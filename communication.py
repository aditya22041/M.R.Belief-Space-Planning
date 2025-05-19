# communication.py

import numpy as np
from scipy.spatial import cKDTree
from config import COMM_RANGE, SHARE_THRESHOLD, REQUEST_PROB, MAX_SHARE

class CommunicationManager:
    """
    Manages communication among robots via one‐way broadcasts and confirmation requests.
    Each robot has an `incoming_messages` queue (list of dicts). On each `step()`:
      1) If a robot’s `new_data_count` ≥ SHARE_THRESHOLD, it sends up to MAX_SHARE of its
         newly explored cells (terrain + victim + belief) to all neighbors within COMM_RANGE.
      2) With probability REQUEST_PROB, a robot asks nearby robots to confirm a random
         “uncertain” cell (P(victim) ∈ (0.1, 0.8)).  That is a small “confirm_request” (not counted).
      3) Finally, `deliver_messages()` processes each queued message:
           - 'share_map'    → integrate terrain/victim/belief data
           - 'confirm_request' → if responder is confident (≥ 0.8) or has seen a victim, send
             back a 'confirm_reply' with that cell’s belief (counted as a long comm).
           - 'confirm_reply'   → reporter merges the provided belief into its own map.
    Communication counts only “share_map” and “confirm_reply” as long/large communications.
    """

    def __init__(self, robots):
        self.robots = robots
        self.kdtree = None

    def update_kdtree(self):
        """
        Build a k‐d tree over current robot positions for fast radius queries.
        """
        positions = np.array([r.pos for r in self.robots])
        self.kdtree = cKDTree(positions)

    def broadcast_share(self, sender_idx):
        """
        If robot `sender_idx` has ≥ SHARE_THRESHOLD new cells, choose up to MAX_SHARE of
        its sensed-but‐not‐yet‐shared cells and bundle that info (terrain, victim, belief)
        into a 'share_map' message for each neighbor within COMM_RANGE.
        """
        sender = self.robots[sender_idx]
        if self.kdtree is None:
            self.update_kdtree()

        idxs = self.kdtree.query_ball_point(sender.pos, r=COMM_RANGE)

        # “mask_new” is every cell that the sender has explored.  For each neighbor, we only
        # send those that the neighbor has not yet explored.
        mask_new = sender.explored_map

        for i in idxs:
            if i == sender_idx:
                continue
            receiver = self.robots[i]
            diff = mask_new & (~receiver.explored_map)
            cells = np.argwhere(diff)
            if cells.size == 0:
                continue

            # If more than MAX_SHARE cells, pick MAX_SHARE at random
            if len(cells) > MAX_SHARE:
                rng = np.random.default_rng()
                chosen = rng.choice(len(cells), size=MAX_SHARE, replace=False)
                cells = cells[chosen]

            # Build payload
            positions = [tuple(x) for x in cells.tolist()]
            terrain_vals = sender.terrain_map[cells[:,0], cells[:,1]].tolist()
            victim_flags = sender.victim_map[cells[:,0], cells[:,1]].tolist()
            beliefs     = sender.confidence_map.belief[cells[:,0], cells[:,1], :].tolist()

            msg = {
                'type':    'share_map',
                'positions': positions,
                'terrain':   terrain_vals,
                'victim':    victim_flags,
                'belief':    beliefs
            }
            receiver.incoming_messages.append(msg)

            # Count as a long communication for both sender and receiver
            sender.comm_count  += 1
            receiver.comm_count += 1

        # Reset counter
        sender.new_data_count = 0

    def broadcast_confirm_request(self, reporter_idx, cell):
        """
        Reporter asks neighbors for confirmation at cell = (i,j).  That is a small message
        and not counted as a “long” communication.
        """
        reporter = self.robots[reporter_idx]
        if self.kdtree is None:
            self.update_kdtree()

        idxs = self.kdtree.query_ball_point(reporter.pos, r=COMM_RANGE)
        for i in idxs:
            if i == reporter_idx:
                continue
            receiver = self.robots[i]
            msg = {
                'type':      'confirm_request',
                'cell':      cell,
                'reporter_id': reporter_idx
            }
            receiver.incoming_messages.append(msg)
        # (We do not increment comm_count for this small request)

    def deliver_messages(self):
        """
        Process each robot’s incoming_messages:
         - 'share_map' → integrate map + merge external belief
         - 'confirm_request' → if the receiver is confident (P(victim)>0.8) or has personally
           detected a victim at that cell, send back a 'confirm_reply' (counted as long comm).
         - 'confirm_reply' → merge that belief vector into the reporter’s belief.
        """
        for receiver in self.robots:
            queue = receiver.incoming_messages
            while queue:
                msg = queue.pop(0)
                typ = msg['type']

                if typ == 'share_map':
                    positions     = msg['positions']
                    terrain_vals  = msg['terrain']
                    victim_flags  = msg['victim']
                    beliefs       = msg['belief']

                    # Update terrain/explored/victim, then fuse belief
                    for idx, (i, j) in enumerate(positions):
                        receiver.terrain_map[i, j] = terrain_vals[idx]
                        receiver.explored_map[i, j] = True
                        if victim_flags[idx]:
                            receiver.victim_map[i, j] = True

                    receiver.confidence_map.merge_external_belief(positions, beliefs)

                elif typ == 'confirm_request':
                    (ci, cj) = msg['cell']
                    rep_id    = msg['reporter_id']
                    p_v       = receiver.confidence_map.belief[ci, cj, 2]
                    # If high confidence OR personally found a victim, reply with belief
                    if (p_v > 0.8) or receiver.victim_map[ci, cj]:
                        belief_vec = receiver.confidence_map.belief[ci, cj, :].tolist()
                        reply = {
                            'type':        'confirm_reply',
                            'cell':        (ci, cj),
                            'belief':      belief_vec,
                            'responder_id': receiver.id
                        }
                        self.robots[rep_id].incoming_messages.append(reply)

                        # Count as a long communication for both parties
                        receiver.comm_count += 1
                        self.robots[rep_id].comm_count += 1

                elif typ == 'confirm_reply':
                    (ci, cj)   = msg['cell']
                    belief_vec = msg['belief']
                    receiver.confidence_map.merge_external_belief([(ci, cj)], [belief_vec])

    def step(self):
        # 1) Rebuild KD‐tree for neighbor lookups
        self.update_kdtree()

        # 2) Broadcast large map‐SHAREs if threshold reached
        for idx, robot in enumerate(self.robots):
            if robot.new_data_count >= SHARE_THRESHOLD:
                self.broadcast_share(idx)

        # 3) Randomly send “confirm_request” for an uncertain cell
        for idx, robot in enumerate(self.robots):
            if np.random.random() < REQUEST_PROB:
                unc = np.argwhere(robot.confidence_map.uncertain_mask)
                if len(unc) > 0:
                    ci, cj = tuple(unc[np.random.randint(len(unc))])
                    self.broadcast_confirm_request(idx, (ci, cj))

        # 4) Deliver all messages (map‐sharing & replies)
        self.deliver_messages()

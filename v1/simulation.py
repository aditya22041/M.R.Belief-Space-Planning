# simulation.py
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_recall_curve, auc, f1_score
from environment import Environment
from robot import Robot
from central_server import CentralServer
from config import (GRID_SIZE, NUM_ROBOTS, MAX_STEPS, SPAWN_NEW_VICTIMS,
                    VICTIM_SPAWN_INTERVAL, VICTIM_SPAWN_PROBABILITY)

def run_simulation():
    print("Initializing simulation..."); start_time=time.time()
    env=Environment(GRID_SIZE); free_cells=np.argwhere(env.reachable); rng=np.random.default_rng()
    start_indices=rng.choice(len(free_cells),size=NUM_ROBOTS,replace=False)
    server=CentralServer((GRID_SIZE,GRID_SIZE),[])
    robots=[Robot(i,tuple(pos),(GRID_SIZE,GRID_SIZE),server) for i,pos in enumerate(free_cells[start_indices])]
    server.robots={r.id:r for r in robots}
    server.update_assignments()

    print(f"Starting simulation: {NUM_ROBOTS} robots, {MAX_STEPS} steps, {len(env.victims)} initial victims.")
    initial_victim=env.victims[0] if env.victims else None; spawn_count=0

    for step in range(MAX_STEPS):
        if step>0 and step%50==0: print(f"--- Step {step}/{MAX_STEPS} ---")
        if SPAWN_NEW_VICTIMS and step>0 and step%VICTIM_SPAWN_INTERVAL==0:
            if np.random.random()<VICTIM_SPAWN_PROBABILITY:
                if env.spawn_new_victim(server.global_explored_map): spawn_count+=1
        env.update_victims()
        for r in robots: r.step(env.terrain_map,env.victim_map)
        server.step(step)

    print("\nSimulation finished. Calculating final statistics...")
    y_true=env.victim_map[env.reachable].flatten(); y_scores=server.belief_map[env.reachable].flatten()
    if np.sum(y_true)>0:
        precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
        pr_auc=auc(recall,precision)
        f1_scores=2*(precision*recall)/(precision+recall+1e-9)
        best_idx=np.argmax(f1_scores[:-1] if len(f1_scores)>1 else f1_scores)
        best_f1,best_p,best_r,best_t=f1_scores[best_idx],precision[best_idx],recall[best_idx],thresholds[best_idx]
    else: pr_auc,best_f1,best_p,best_r,best_t=0,0,0,0,0
    
    print("Generating and saving plots...")

    # Plot 1: Exploration Map
    fig1,ax1=plt.subplots(figsize=(12,12));
    ax1.set_title("Exploration Map & Robot Assignments",fontsize=16,weight='bold')
    terrain_colored=np.zeros((GRID_SIZE,GRID_SIZE,3))
    terrain_colored[env.terrain_map==0]=[0.1,0.1,0.1]; terrain_colored[env.terrain_map==1]=[0.5,0.5,0.5]
    ax1.imshow(terrain_colored,origin='lower')
    assignment_map=server.robots[0].assignment_map
    if assignment_map is not None:
        assignment_colored=plt.get_cmap('gist_rainbow')(assignment_map/NUM_ROBOTS)
        assignment_colored[~server.global_explored_map]=0
        assignment_colored[...,3]=0.5
        ax1.imshow(assignment_colored,origin='lower')
    fig1.savefig("exploration_map.png")

    # Plot 2: Victim Belief Heatmap with True Paths
    fig2,ax2=plt.subplots(figsize=(12,12));
    ax2.set_title("Victim Belief Heatmap & True Victim Paths",fontsize=16,weight='bold')
    ax2.imshow(1-env.terrain_map,cmap='gray',origin='lower',alpha=0.2)
    im=ax2.imshow(server.belief_map,cmap='hot',origin='lower',vmin=0,vmax=1)
    fig2.colorbar(im,ax=ax2,fraction=0.046,pad=0.04).set_label('Confirmed Belief Score')
    colors=plt.cm.cool(np.linspace(0,1,len(env.victims)))
    for i,victim in enumerate(env.victims):
        path=np.array(victim.history)
        ax2.plot(path[:,1],path[:,0],color=colors[i],linewidth=2,alpha=0.7)
        ax2.scatter(path[-1,1],path[-1,0],c=[colors[i]],marker='*',s=250,edgecolor='white',label=f"Victim {victim.id}")
    ax2.legend()
    fig2.savefig("heatmap_with_victims.png")

    # Plot 3: Statistics Summary
    fig3,ax3=plt.subplots(figsize=(10,7)); ax3.axis('off'); ax3.set_title("Performance Summary",fontsize=16,weight='bold')
    stats_text=(f"Total Simulation Time: {time.time()-start_time:.2f} seconds\n"
                  f"{'-'*45}\n"
                  f"Explored Area: {np.sum(server.global_explored_map)/np.sum(env.reachable)*100:.2f}%\n"
                  f"Communication Events: {server.comm_count} messages\n"
                  f"Total Victims at End: {len(env.victims)}\n"
                  f"{'-'*45}\n"
                  f"Precision-Recall AUC: {pr_auc:.3f}\n"
                  f"Best F1-Score: {best_f1:.3f} (at threshold={best_t:.2f})\n"
                  f"  > Precision: {best_p:.3f}\n"
                  f"  > Recall:    {best_r:.3f}")
    ax3.text(0.5,0.5,stats_text,ha='center',va='center',fontsize=12,fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=1",fc='ivory'))
    fig3.savefig("statistics_summary.png")
    
    print("\nPlots saved to PNG files. Now attempting to display them...")
    plt.show()

if __name__ == "__main__":
    run_simulation()
import numpy as np
from matplotlib import pyplot as plt
def Plot_Results_new():
    for a in range(2):
        Eval = np.load('Eval_Epoch.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'MD', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'ONE-NORM', 'TWO-NORM', 'MAPE']
        # Colors for each category
        colors = [ '#929591']
        Terms_Bar = ['MRSCO-HDA-ADDNet']
        Graph_ = [0,1,2,3,4,5,6,7,8,9]
        for m in range(len(Graph_)):
            Graph = Eval[:, 4, Graph_[m]]

            bar_width = 0.25
            x = np.arange(5)
            plt.figure(figsize=(7, 6))

            # Plot bars for each category (Algorithm)
            plt.bar(x + bar_width, Graph, width=bar_width, label='MRSCO-HDA-ADDNet', color=colors[0])

            # Add PROPOSED line plot
            plt.plot(x + bar_width, Graph, color='red', marker='o', linestyle='-', linewidth=2)

            # Customizations
            plt.xticks(x + bar_width, ['50', '100', '150','200', '250'], fontsize=10,
                       fontweight='bold')
            plt.xlabel('Epochs', fontsize=12, fontweight='bold')
            plt.ylabel(Terms[Graph_[m]], fontsize=12, fontweight='bold')

            # Remove axes outline
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)

            # Custom Legend with Dot Markers and Line Marker
            dot_markers = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
                           colors]
            line_marker = plt.Line2D([0], [0], color='red', marker='o', linestyle='-', linewidth=2)
            plt.legend(dot_markers , Terms_Bar , fontsize=10,
                       loc='upper center', bbox_to_anchor=(1.05, 0.5), frameon=False, ncol=1)

            # Add gridlines for y-axis only
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Adjust layout for legend space outside the graph
            plt.tight_layout()

            plt.savefig(f"./Results/New_Plot_Dataset_{a+1}_{Terms[Graph_[m]]}_Proposed.png", dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(1)
            plt.close()

if __name__ == '__main__':
    Plot_Results_new()

import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


mpl.use("pgf")
mpl.rcParams.update({
	
	# LaTex required params
	"pgf.texsystem": "pdflatex",
	"font.family": "serif",
	"text.usetex": True,
	"pgf.rcfonts": False,
	
})


def figures_exist(subpath, model_dir, figure_name, *file_types):
	for file_type in file_types:
		if not os.path.exists(os.path.join(subpath, model_dir, figure_name + "." + file_type)):
			return False
	
	return True


def save_figures(subpath, model_dir, figure_name, *file_types):
	for file_type in file_types:
		plt.savefig(os.path.join(subpath, model_dir, figure_name + "." + file_type), bbox_inches="tight")


def main():
	agents_dir = ".\\agents"
	regenerate_existing = True
	
	plt.style.use("seaborn-poster")
	
	# Look through all the environments
	for subdir in os.listdir(agents_dir):
		subpath = os.path.join(agents_dir, subdir)
		
		models = []
		num_models = len(os.listdir(subpath))
		model_data = np.zeros(((num_models - 1) * 3, 250))
		
		# Look through each model
		model_num = 0
		for model_dir in os.listdir(subpath):
			if os.path.isfile(os.path.join(subpath, model_dir)):
				continue
			
			print(f"Generating charts for {model_dir}")
			
			models.append(model_dir + "_Rewards")
			models.append(model_dir + "_Losses")
			models.append(model_dir + "_ValueEstimates")
			
			data_file = os.path.join(subpath, model_dir, "train_data.npz")
			
			with np.load(data_file) as train_data:
				rewards = train_data["rewards"]
				losses = train_data["losses"]
				preds = train_data["preds"]
				
				model_data[model_num * 3] = rewards
				model_data[model_num * 3 + 1] = losses
				model_data[model_num * 3 + 2] = preds
				
				# Calculate reward averages
				rewards_avg = np.zeros(len(rewards))
				for i in range(len(rewards)):
					rewards_avg[i] = rewards[0 if i - 100 < 0 else i - 100: i+1].mean()
					
				# Find first episode where true loss was counted
				loss_first = 1
				while losses[loss_first] == losses[0]:
					loss_first += 1
				
				# Create the rewards plot
				if not figures_exist(subpath, model_dir, "rewards", "png", "svg", "pgf") or regenerate_existing:
					fig, ax = plt.subplots()
					ax.plot(rewards)
					ax.plot(rewards_avg, linewidth=4, dashes=(2, 1))
					ax.set_ylabel("Reward", fontsize=64)
					ax.set_xlabel("Episode", fontsize=64)
					plt.xticks(fontsize=54)
					plt.yticks(fontsize=54)
					
					# Quick check for setting y bounds based on environment
					if subdir == "CartPole-v1":
						ax.set_ylim([0, 510])
					elif subdir == "LunarLander-v2":
						ax.set_ylim([-500, 310])
					
					save_figures(subpath, model_dir, "rewards", "png", "svg", "pgf")
					plt.close(fig)
				
				# Create the losses
				if not figures_exist(subpath, model_dir, "losses", "png", "svg", "pgf") or regenerate_existing:
					fig, ax = plt.subplots()
					ax.plot(range(loss_first, len(losses)), losses[loss_first:])
					
					ax.set_ylabel("Loss", fontsize=64)
					ax.set_xlabel("Episode", fontsize=64)
					plt.xticks(fontsize=54)
					plt.yticks(fontsize=54)
					
					save_figures(subpath, model_dir, "losses", "png", "svg", "pgf")
					plt.close(fig)
				
				# Create the prediction values plot
				if not figures_exist(subpath, model_dir, "value_estimates", "png", "svg", "pgf") or regenerate_existing:
					fig, ax = plt.subplots()
					ax.plot(preds)
					
					ax.set_ylabel("Q-Value Estimate", fontsize=64)
					ax.set_xlabel("Episode", fontsize=64)
					plt.xticks(fontsize=54)
					plt.yticks(fontsize=54)
					
					save_figures(subpath, model_dir, "value_estimates", "png", "svg", "pgf")
					plt.close(fig)
			
			model_num += 1
		
		np.savetxt(
			os.path.join(subpath, "data.csv"),
			model_data.transpose(),
			fmt="%.5f",
			delimiter=",",
			header=",".join(models)
		)


if __name__ == "__main__":
	main()

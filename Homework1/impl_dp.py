#Author: Sruti Bhagavatula (Jan 2022)

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import math


def get_dataset():
	#  Read in dataset and convert necessary fields
	dataset = np.loadtxt("adult-trunc.data", delimiter=", ", dtype='str')
	dataset = dataset.tolist()
	for i in range(len(dataset)):
		dataset[i][0] = int(dataset[i][0])
		dataset[i][12] = int(dataset[i][12])
	return dataset


# Dataset statistics
def compute_avg(dataset, col_idx):
	# Arguments
	# dataset: dataset over which to compute the statistic
	# col_idx: index of column (or attribute) to average over

	col_vals_continuous = np.asarray([row[col_idx] for row in dataset])
	return np.mean(col_vals_continuous, axis=0)


def compute_max(dataset, col_idx):
	# Arguments
	# dataset: dataset over which to compute the statistic
	# col_idx: index of column (or attribute) over which to compute the maximum

	col_vals_continuous = np.asarray([row[col_idx] for row in dataset])
	return np.max(col_vals_continuous)


# ------------------------------------------------
# Other functions
def compute_global_sensitivity(dataset, func, col_idx, norm):
	# Arguments:
	# dataset: released dataset
	# func: function to be made differentially private. func can be called like "func(<args>)"
	# col_idx: index of the column corresponding to the attribute in question
	# norm: "l1" or "l2"
	# Compute f(D) on the original dataset
	f_ds = func(dataset, col_idx)
	gs = 0

	#TODO: Compute and return the global sensitivity of the function according to the norm

	for i in range(len(dataset)):
		ds_prime = dataset.copy()
		ds_prime.pop(i) #ds_prime contains a copy of the database with row i removed

		#TODO: Compute the difference between f(D) and f(D') according to norm. You may use the math library if you'd like
		if norm == "l1":
			new_fds = func(ds_prime, col_idx)
			diff = abs(new_fds - f_ds)
			gs = max(gs, diff)

		if norm == "l2":
			new_fds = func(ds_prime, col_idx)
			diff = math.sqrt((new_fds - f_ds) ** 2)
			gs = max(gs, diff)

	return gs
	#TODO: Return global sensitivity


def add_gaussian_noise(dataset, std):
	sigma = std ** 2
	val_gauss = np.random.normal(loc=0, scale=sigma)
	# print(val_gauss)
	return val_gauss
	#	Arguments:
	#	dataset: released dataset
	#	std: standard deviation of the distribution (NOT THE VARIANCE)

	#TODO: Return random Gaussian noise using np.random.normal...


def add_laplacian_noise(dataset, b):
	val_laplace = np.random.laplace(loc=0, scale=b)
	return val_laplace
	# Arguments:
	#	dataset: released dataset
	#	b: scale of the distribution

	#TODO: Return random Laplace noise using np.random.laplace...
		
def plot_epsilon(dataset, func, col_idx, noise, eps_range, norm):
	#Arguments:
	#	dataset: released dataset
	#	col_idx: index of the column corresponding to the attribute in question
	# 	noise: "gaussian" or "laplace"
	# 	eps_range: array of epsilon values for which to compute accuracy
	#	norm: "l1" or "l2"

	#Compute f(D) on the original database
	f_ds = func(dataset, col_idx)

	#Compute global sensitivity of the statistic function
	GS = compute_global_sensitivity(dataset, func, col_idx, norm) 

	all_diffs = [] #empty list to store all the differences between f(D) and f(D')

	if noise == "gaussian":
		delta = 1 #fixed delta for Gaussian mechanism

	for e in eps_range:
		if noise == "laplace":
			b = GS / e
			M = f_ds + add_laplacian_noise(dataset, b)
			# M = f(D) + noise (for laplace) l1 norm
			#  TODO: Compute the randomized sanitization function M by adding Laplacian noise

		elif noise == "gaussian":
			sigma = np.sqrt((2 * np.log(1.25 / delta) * (GS ** 2)) / (e ** 2))
			# print(sigma)
			M = f_ds + add_gaussian_noise(dataset, sigma)  # noise for gaussian l2 norm
			#  TODO: Compute the randomized sanitization function M by adding Gaussian noise

		all_diffs.append(abs(M-f_ds)) #Compute difference between true f(D) and M for plotting (accuracy)

	#Plot epsilon values vs. accuracy of function
	plt.plot(eps_range, all_diffs)
	plt.savefig('plot-{0}-{1}.png'.format(func.__name__, noise))
	plt.clf()


def main():
	dataset = get_dataset()

	# Measure accuracy for avg. age
	print("Plotting accuracies for the compute_avg function for both Gaussian and Laplace noise.")
	plot_epsilon(dataset, compute_avg, 0, noise="gaussian", eps_range=np.arange(0.1,1.0, 0.01), norm="l2")

	plot_epsilon(dataset, compute_avg, 0, noise="laplace", eps_range=np.arange(0.1, 5, 0.01), norm="l1")


	# Measure accuracy for hours-per-week
	print("Plotting accuracies for the compute_max function for both Gaussian and Laplace noise.")
	plot_epsilon(dataset, compute_max, 12, noise="gaussian", eps_range=np.arange(0.1,1.0, 0.01), norm="l2")

	plot_epsilon(dataset, compute_max, 12, noise="laplace", eps_range=np.arange(0.1, 5, 0.01), norm="l1")


if __name__ == '__main__':
	main()	

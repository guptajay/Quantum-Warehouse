import numpy as np
from warehouse.envs.constants import packages, n_packages

class Package():
	''' Descibe any package.
	'''
	
	def __init__(self, package_id, frequency,weight, spot = None):
		''' The tuple (package_id,frequency,weight) should be fixed! 
		'''
	
		# Important characteristics
		self.id  = package_id
		self.frequency = frequency
		self.weight = weight
		
		# Internal states
		self._is_stored = False # if it is on a spot
		self._spot = spot
		self._waiting_time = 0 # the time it has been waiting to be put on a spot
		
	def cost(self,distance):
		''' Compute the cost of placing this object at the given distance
		'''
		pass
		
	
	def __repr__(self):
		return 'Package(%r,%r)' % (self.frequency, self.weight)

	def __str__(self):
		return 'Package (f:%s, w:%s), stored : %s, waiting for %s' % \
			   (self.frequency, self.weight, self._is_stored, self._waiting_time)
		


def generate_packages(number=1):
	'''
	Generate an incoming package to store. The object characteristics should follow a fixed distribution

	# Args
	 - Number : number of package to generate

	# Output
	 - List of generated packages
	'''

	# Draw number of package to generate per type based on frequency.
	generated_indices = np.random.choice( [i for i in range(n_packages)], number, p = [package[1] for package in packages] )

	# Add packages to the list and return
	return [ Package(*packages[index]) for index in generated_indices ]
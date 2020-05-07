
# A really low-tech method for finding the input to the minimum of a 1D convex function
def convex_1d_opt(fun, range_low, range_high, num_iter=100, eps=1e-5):
	
	# Bisection method with emperical derivative
	
	# Non-analytic derivative
	def grad(x):
		numer = fun(x + eps) - fun(x - eps)
		denom = 2*eps
		
		return numer / eps

	for _ in range(num_iter):
		guess_x = (range_low + range_high) / 2
		
		deriv = grad(guess_x)
		if deriv < 0:
			# optimum is higher
			range_low = guess_x
		elif deriv > 0:
			# optimum is lower
			range_high = guess_x
		else:
			# we are at the optimum
			break
			
	return guess_x


if __name__ == '__main__':
	
	# Simple test:
	
	def fun(x):
		return x ** 2 - 2 * x
	print('{:.4f}'.format(convex_1d_opt(fun, -5, 10)))
	
	def fun(x):
		return 1 - (1 - x ** 2) ** 0.5
	print('{:.4f}'.format(convex_1d_opt(fun, -1, 1)))

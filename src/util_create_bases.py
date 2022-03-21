from imports import *

def create_legendre_basis(deg):
    l = [None]*deg
    for i in range(deg):
        l[i] = legendre.Legendre([0]*i + [1])
    global legendre_basis
    def legendre_basis(x):
        basis_vals = np.zeros((len(x), deg))
        for i in range(deg):
            basis_vals[:,i] = l[i](x).flatten()
        return torch.tensor(basis_vals)
    return legendre_basis

def create_random_linear_basis(num_bases):
    slopes = np.random.uniform(low=-5.0, high=5.0, size=num_bases)
    intercepts = np.random.uniform(low=-5.0, high=5.0, size=num_bases)
    global random_linear_basis
    def random_linear_basis(x):
        basis_vals = np.zeros((len(x), num_bases))
        for i in range(num_bases):
            basis_vals[:,i] = slopes[i] * x.flatten() + intercepts[i]
        return torch.tensor(basis_vals)
    return random_linear_basis







# # the below is attempting to create basis from vector
# # but we decided it's fine to have

# def find_closest_idx(A, x):
#     left, right = 0, len(A) - 1
#     while left < right:
#         mid = (left + right) / 2
#         if x - A[mid] > A[mid + 1] - x:
#             left = mid + 1
#         else:
#             right = mid
#     return left
#
#
# def create_basis(basis_vals, x_vals):
#     # basis vals defined on x_vals
#     # takes floor function
#     # currently for 1d datasets
#     # assume x_vals is the same for every basis function
#     assert len(basis_vals.shape) == 2
#     num_bases, num_points = basis_vals.shape
#     assert len(x_vals) == num_points
#
#     range = x_vals[-1] - x_vals[0]
#
#     def basis(x):
#         try:
#             iter(x)
#             ans = np.zeros((len(x), num_bases))
#             for j in range(len(x)):
#                 # make sure that tested x value isn't too out of range...
#                 assert x[j] > x_vals[0] - range/3 and x[j] < x_vals[-1] + range/3
#                 i = find_closest_idx(x_vals, x[j])
#                 ans[j] = basis_vals[:, i]
#
#         except TypeError:
#             # dealing with scalar
#             ans = np.zeros(len(x))
#             assert x > x_vals[0] - range/3 and x < x_vals[-1] + range/3
#             i = find_closest_idx(x_vals, x)
#             ans = basis_vals[:, i]
#
#         return ans
#
#     return basis

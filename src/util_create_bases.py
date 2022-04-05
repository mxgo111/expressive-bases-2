from imports import *
from sklearn.kernel_approximation import RBFSampler

torch.set_default_tensor_type(torch.DoubleTensor)

def cubic(x):
    return torch.pow(x, 3.0)

def sine(x):
    return torch.sin(x)

def xsinx(x):
    return torch.multiply(x, torch.sin(x))

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

# needs to be deleted
def create_adv_basis(num_bases):
    slopes = np.random.uniform(low=-5.0, high=5.0, size=num_bases)
    intercepts = np.random.uniform(low=-5.0, high=5.0, size=num_bases)
    global random_adv_basis
    def random_adv_basis(x):
        basis_vals = np.zeros((len(x), num_bases))
        for i in range(num_bases-1):
            basis_vals[:,i] = slopes[i] * x.flatten() + intercepts[i]
        basis_vals[:, -1] = torch.pow(x.flatten(), 3.0) # set one basis to be cubic
        return torch.tensor(basis_vals)
    return random_adv_basis


def create_fourier_basis(num_bases, omega_scale=1.0):
    # source:
    # https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/#kimeldorf1971some
    omegas = np.random.normal(loc=0, scale=omega_scale, size=num_bases)
    bs = np.random.uniform(low=0.0, high=np.pi*2, size=num_bases)
    # a = 1/np.sqrt(np.sqrt(2 * np.pi) / omega_scale)
    a = 1
    global random_fourier_basis
    def random_fourier_basis(x):
        basis_vals = np.zeros((len(x), num_bases))
        for i in range(num_bases):
            basis_vals[:,i] = np.sqrt(2)/(np.sqrt(num_bases)) * a * np.cos(omegas[i] * x.flatten() + bs[i])
        return torch.tensor(basis_vals)
    return random_fourier_basis

def create_one_basis_match(create_bases_function, correct_basis_function, num_bases):
    """
    Creates a set of bases with one of the bases being correct
    """
    basis_function = create_bases_function(num_bases)
    global one_basis_match
    def one_basis_match(x):
        basis_vals = basis_function(x)
        basis_vals[:,-1] = torch.tensor(correct_basis_function(x))
        return torch.tensor(basis_vals)
    return one_basis_match

# also needs to be deleted
def create_fourier_basis_one_match(num_bases):
    omegas = np.random.randn(num_bases)
    bs = np.random.uniform(low=0.0, high=np.pi*2, size=num_bases)
    global random_fourier_basis_one_match
    def random_fourier_basis_one_match(x):
        basis_vals = np.zeros((len(x), num_bases))
        for i in range(num_bases):
            basis_vals[:,i] = np.sqrt(2)/np.sqrt(num_bases) * np.cos(omegas[i] * x.flatten() + bs[i])
        basis_vals[:,-1] = torch.pow(x.flatten(), 3.0) # cubic
        return torch.tensor(basis_vals)
    return random_fourier_basis_one_match

# new rffs based on sklearn
def create_rffs_sklearn(num_bases, length_scale):
    rbf_features = RBFSampler(gamma=1/(2 * (length_scale ** 2)), n_components=num_bases, random_state=1)
    global rffs_sklearn
    def rffs_sklearn(x):
        basis_vals = rbf_features.fit_transform(x.reshape(-1, 1))
        return torch.tensor(basis_vals)
    return rffs_sklearn

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

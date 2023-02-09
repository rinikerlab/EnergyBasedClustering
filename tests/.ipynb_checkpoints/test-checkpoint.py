import sys
sys.path.append('../source')

from EBC import EBC
import numpy as np
   
def _test_initialization(samples, energies):
    ebc = EBC(n_clusters=10) 
    ebc.fit(samples, energies)
    return ebc

def _test_sparse(samples, energies):    
    for n in [1, 10, 100]:
        for knn in [4, 16, 64]:
            for radius in [0.4, 0.8, 1.2, 2]:
                for T in [1, 10, 100, 1000]:                
                    ebc_sparse = EBC(temperature=T, n_clusters=n, proto_radius=radius, knn=knn, use_sparse=True) 
                    ebc_sparse.fit(samples, energies)
                    ebc = EBC(temperature=T, n_clusters=n, proto_radius=radius, knn=knn, use_sparse=False) 
                    ebc.fit(samples, energies)
                    assert np.allclose(ebc.pi, ebc_sparse.pi)
                    assert np.allclose(ebc.diffusion_matrix, ebc.diffusion_matrix_sp.toarray())
                    
def _test_plotting(ebc):
    ebc.plot_graph()
    ebc.show()
    ebc.hierarchical()
                    
if __name__ == "__main__":
    samples_full = np.load('../source/examples/data/10_well_trajectory_5.npy')
    energies_full = np.load('../source/examples/data/10_well_potential_5.npy') 
    samples_10K = samples_full[::100]
    energies_10K = energies_full[::100]
    ebc = _test_initialization(samples_10K, energies_10K)
    _test_sparse(samples_10K, energies_10K)
    _test_plotting(ebc)
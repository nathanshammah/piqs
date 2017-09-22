"""
Tests for Permutation invariance class
"""
import numpy as np
from numpy.testing import assert_, run_module_suite, assert_raises, assert_array_equal

from pim import num_dicke_states, irreducible_dim, Pim, tau_column

class TestPim:
    """
    A test class for the Permutation invariance matrix generation
    """
    def test_num_dicke_states(self):
        """
        Tests the `num_dicke_state` function
        """
        N_list = [1, 2, 3, 4, 5, 6, 9, 10, 20, 100, 123]
        dicke_states = [num_dicke_states(i) for i in N_list]

        assert_array_equal(dicke_states, [2, 4, 6, 9, 12, 16, 30, 36, 121, 2601, 3906])

        N = -1
        assert_raises(ValueError, num_dicke_states, N)

        N = 0.2
        assert_raises(ValueError, num_dicke_states, N)
    
    
    # Nathan testing STARTS
    
    def test2_num_dicke_states(self):
        """
        Tests again the `num_dicke_states` function
        """
        nds_list = []
        for N in range (1, 10):
            nds_list.append(num_dicke_states(N)) 
        nds_true = [2,4,6,9,12,16,20,25,30]
        if nds_list != nds_true:
            raise ValueError("Incorrect number of Dicke states calculated")
        pass

    def num_two_level(self):
        """
        Tests the `num_two_level` function, which gives N, the number of two level systems, if it given the number of Dicke states. 
        """

        for N in range (1, 10):
            nds = num_dicke_states(N)
            NN = num_two_level(nds)
            if N != NN:
                raise ValueError("Incorrect N calculated from num_two_level, with ds = {}".format(nds))
        pass 
    
    def test_num_dicke_ladders(self):
        """
        Tests the `num_dicke_ladders` function
        """
        
        ndl = []
        ndl_true = [1, 2, 2, 3, 3, 4, 4, 5, 5]
        for N in range (1, 10):
            ndl.append(num_dicke_ladders(N))
        
        if ndl != ndl_true:
            raise ValueError("Incorrect number of Dicke ladders calculated")
        pass 

    # Nathan testing ENDS

    
    def test_irreducible_dim(self):
        """
        Test the irreducible dimension function
        """
        pass

    def test_isdicke(self):
        """
        Tests the `isdicke` function
        """
        N1 = 1
        g0=.01
        nth=.01
        gP=g0*nth
        gL=g0*(0.1+nth)
        gS= 0.1
        gD= 0.1

        pim1 = Pim(N1, gS, gL, gD, gP)

        test_indices1 = [(0, 0), (0, 1), (1, 0), (-1, -1), (2, -1)]
        dicke_labels = [pim1.isdicke(x) for x in test_indices1]

        N2 = 4
        
        pim2 = Pim(N2, gS, gL, gD, gP)
        test_indices2 = [(0, 0), (4, 4), (2, 0), (1, 3), (2, 2)]
        dicke_labels = [pim2.isdicke(x) for x in test_indices2]

        assert_array_equal(dicke_labels, [True, False, True, False, True])

if __name__ == "__main__":
    run_module_suite()
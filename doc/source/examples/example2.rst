============================================================
PIQS model
============================================================
The intravoxel incoherent motion (IVIM) model describes diffusion
and perfusion in the signal acquired with a diffusion MRI sequence
that contains multiple low b-values. The IVIM model can be understood
as an adaptation of the work of Stejskal and Tanner [Stejskal65]_
in biological tissue, and was proposed by Le Bihan [LeBihan84]_.
The model assumes two compartments: a slow moving compartment,
where particles diffuse in a Brownian fashion as a consequence of thermal
energy, and a fast moving compartment (the vascular compartment), where
blood moves as a consequence of a pressure gradient. In the first compartment,
the diffusion coefficient is :math:`\mathbf{D}` while in the second compartment, a
pseudo diffusion term $\mathbf{D^*}$ is introduced that describes the
displacement of the blood elements in an assumed randomly laid out vascular
network, at the macroscopic level. According to [LeBihan84]_,
:math:`\mathbf{D^*}` is greater.

The IVIM model expresses the MRI signal as follows:

 .. math::
    S(b)=S_0(fe^{-bD^*}+(1-f)e^{-bD})

In the following example we show how to fit the IVIM model on a
diffusion-weighteddataset and visualize the diffusion and pseudo
diffusion coefficients. First, we import all relevant modules:

.. code-block:: python
  
    import matplotlib.pyplot as plt
    from dipy.reconst.ivim import IvimModel
    from dipy.data.fetcher import read_ivim

.. figure:: images/piqs_logo.png
   :align: center


References:

.. [Stejskal65] Stejskal, E. O.; Tanner, J. E. (1 January 1965).
                "Spin Diffusion Measurements: Spin Echoes in the Presence
                of a Time-Dependent Field Gradient". The Journal of Chemical
                Physics 42 (1): 288. Bibcode: 1965JChPh..42..288S.
                doi:10.1063/1.1695690.

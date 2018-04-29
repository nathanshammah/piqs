====================================
Spin ensembles: Negative temperature
====================================
We consider a system of two two-level systems (TLSs) :math:`N_1` and :math:`N_2` with identical frequency :math:`\omega_0` with collective pumping and collective emission at identical rates, :math:`\gamma_\text{CE} = (1+\bar{n})\gamma_0` and :math:`\gamma_\text{CP}=\bar{n}\gamma_0`, respectively, with :math:`\bar{n}=\frac{1}{e^{\hbar\omega_0/k_\mathrm{B}T}-1}` and 

.. math::
	
	\dot{\rho} = -i\lbrack \omega_{0}\left(J_z^{(1)}+J_z^{(2)}\right),\rho \rbrack
	+\frac{\gamma_\text {CE}}{2}\mathcal{L}_{J_{-}^{(1)}+ J_{-}^{(2)}}[\rho]
	+\frac{\gamma_\text {CP}}{2}\mathcal{L}_{J_{+}^{(1)}+J_{+}^{(2)}}[\rho]
	

Hama *et al.* have shown in Ref. [1] that for :math:`N_1<N_2`, if the system is initialized in the state :math:`|{\psi_0}\rangle=|{\downarrow\cdots\downarrow}\rangle_1\otimes|{\uparrow\cdots\uparrow}\rangle_2`, the system relaxes to a steady state for which the first subsystem is excited, i.e. :math:`\langle J_z^{(1)}(\infty)\rangle>0` and for some parameters  :math:`\frac{\langle J_z^{(1)}(\infty)\rangle}{(N_1/2)}\rightarrow 0.5`, also in the limit of zero temperature, :math:`T\rightarrow 0`.  

Notice that :math:`\mathcal{L}_{J_{-}^{(1)}+ J_{-}^{(2)}}[\rho]\neq \mathcal{L}_{J_{-}^{(1)}}[\rho]+\mathcal{L}_{ J_{-}^{(2)}}[\rho]`, which is a case treated in Ref. [2] two obtain syncronized ensembles of atoms. 

Here we explore what happens to the master equation of Eq. (1) one adds also collective and local terms relative to single ensembles, 

.. math::

	\dot{\rho} =
	-i\lbrack \omega_{0}\left(J_z^{(1)}+J_z^{(2)}\right),\rho \rbrack
	+\frac{\gamma_\text{CE}}{2}\mathcal{L}_{J_{-}^{(1)}+ J_{-}^{(2)}}[\rho]
	+\frac{\gamma_\text{CP}}{2}\mathcal{L}_{J_{+}^{(1)}+J_{+}^{(2)}}[\rho]
	+ \frac{\gamma_\text{CEi}}{2}\mathcal{L}_{J_{-}^{(1)}}[\rho]
	+\frac{\gamma_\text{CEi}}{2}\mathcal{L}_{J_{-}^{(2)}}[\rho]

	+\sum_{n}^{N_1}\frac{\gamma_\text{E}}{2}\mathcal{L}_{J_{-,n}^{(1)}}[\rho]+\frac{\gamma_\text{D}}{2}\mathcal{L}_{J_{z,n}^{(1)}}[\rho]+\sum_{n}^{N_2}\frac{\gamma_\text{E}}{2}\mathcal{L}_{J_{-,n}^{(2)}}[\rho]+\frac{\gamma_\text{D}}{2}\mathcal{L}_{J_{z,n}^{(2)}}[\rho]


where :math:`\gamma_\text {CEi}` is the rate of superradiant decay for the individual ensembles of TLSs, :math:`\gamma_\text{E}` and :math:`\gamma_\text{D}` are the rates of local emission and dephasing.

Firstly, we will show how the collective dynamics of Eq. (1) can be investigated in a simple way using QuTiP's [3] :math:`\texttt{jmat}` function, which defines collective spins for maximally symmetric states in a Hilbert space of dimension :math:`N_i+1`.

Secondly, we will exploit the permutational invariance of the local processes in Eq. (2) to investigate the exact dynamics using the Dicke basis, :math:`\rho = \sum_{j,m,m'}p_{j,m,m'}|j,m\rangle\langle j,m'|` [4]. We will do so numerically using the PIQS library [5]. 

In the following we might use in plots thefollowing equivalent notation :math:`\gamma_\text {CE}=\gamma_\Downarrow`,
:math:`\gamma_\text {CP}=\gamma_\Uparrow`, :math:`\gamma_\text {E}=\gamma_\downarrow`, :math:`\gamma_\text {P}=\gamma_\uparrow`, and 
:math:`\gamma_\text {D}=\gamma_\phi`.

References:

.. [1] 

.. [2]

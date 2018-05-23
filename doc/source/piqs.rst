Documentation
============

.. raw:: html

	<p>Permutational Invariant Quantum Solver (PIQS)</p>
	<p>This module calculates the Liouvillian for the dynamics of ensembles of
	identical two-level systems (TLS) in the presence of local and collective
	processes by exploiting permutational symmetry and using the Dicke basis.</p>
	<dl class="class">
	<dt id="piqs.dicke.Dicke">
	<em class="property">class </em><code class="descclassname">piqs.dicke.</code><code class="descname">Dicke</code><span class="sig-paren">(</span><em>N</em>, <em>hamiltonian=None</em>, <em>emission=0.0</em>, <em>dephasing=0.0</em>, <em>pumping=0.0</em>, <em>collective_emission=0.0</em>, <em>collective_dephasing=0.0</em>, <em>collective_pumping=0.0</em><span class="sig-paren">)</span>
	<dd><p>Bases: <a class="reference external" href="https://docs.python.org/3/library/functions.html#object" title="(in Python v3.6)"><code class="xref py py-class docutils literal notranslate"><span class="pre">object</span></code></a></p>
	<p>The Dicke class which builds the Lindbladian and Liouvillian matrix.</p>
	<p class="rubric">Example</p>
	<div class="highlight-default notranslate"><div class="highlight"><pre><span></span><span class="gp">&gt;&gt;&gt; </span><span class="kn">from</span> <span class="nn">piqs</span> <span class="k">import</span> <span class="n">Dicke</span><span class="p">,</span> <span class="n">jspin</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">N</span> <span class="o">=</span> <span class="mi">2</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">jx</span><span class="p">,</span> <span class="n">jy</span><span class="p">,</span> <span class="n">jz</span> <span class="o">=</span> <span class="n">jspin</span><span class="p">(</span><span class="n">N</span><span class="p">)</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">jp</span> <span class="o">=</span> <span class="n">jspin</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="s2">&quot;+&quot;</span><span class="p">)</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">jm</span> <span class="o">=</span> <span class="n">jspin</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="s2">&quot;-&quot;</span><span class="p">)</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">ensemble</span> <span class="o">=</span> <span class="n">Dicke</span><span class="p">(</span><span class="n">N</span><span class="p">,</span> <span class="n">emission</span><span class="o">=</span><span class="mf">1.</span><span class="p">)</span>
	<span class="gp">&gt;&gt;&gt; </span><span class="n">L</span> <span class="o">=</span> <span class="n">ensemble</span><span class="o">.</span><span class="n">liouvillian</span><span class="p">()</span>
	</pre></div>
	</div>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>hamiltonian</strong> -- <p>A Hamiltonian in the Dicke basis.</p>
	<p>The matrix dimensions are (nds, nds),
	with nds being the number of Dicke states.
	The Hamiltonian can be built with the operators
	given by the <cite>jspin</cite> functions.</p>
	</li>
	<li><strong>emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent emission coefficient (also nonradiative emission).
	default: 0.0</li>
	<li><strong>dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Local dephasing coefficient.
	default: 0.0</li>
	<li><strong>pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective (superradiant) emmission coefficient.
	default: 0.0</li>
	<li><strong>collective_pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective dephasing coefficient.
	default: 0.0</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.N">
	<code class="descname">N</code><a class="headerlink" href="#piqs.dicke.Dicke.N" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>int</em> -- The number of two-level systems.</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.hamiltonian">
	<code class="descname">hamiltonian</code><a class="headerlink" href="#piqs.dicke.Dicke.hamiltonian" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>:class: qutip.Qobj</em> -- A Hamiltonian in the Dicke basis.</p>
	<p>The matrix dimensions are (nds, nds),
	with nds being the number of Dicke states.
	The Hamiltonian can be built with the operators given
	by the <cite>jspin</cite> function in the &quot;dicke&quot; basis.</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.emission">
	<code class="descname">emission</code><a class="headerlink" href="#piqs.dicke.Dicke.emission" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Incoherent emission coefficient (also nonradiative emission).
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.dephasing">
	<code class="descname">dephasing</code><a class="headerlink" href="#piqs.dicke.Dicke.dephasing" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Local dephasing coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.pumping">
	<code class="descname">pumping</code><a class="headerlink" href="#piqs.dicke.Dicke.pumping" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Incoherent pumping coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.collective_emission">
	<code class="descname">collective_emission</code><a class="headerlink" href="#piqs.dicke.Dicke.collective_emission" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective (superradiant) emmission coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.collective_dephasing">
	<code class="descname">collective_dephasing</code><a class="headerlink" href="#piqs.dicke.Dicke.collective_dephasing" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective dephasing coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.collective_pumping">
	<code class="descname">collective_pumping</code><a class="headerlink" href="#piqs.dicke.Dicke.collective_pumping" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective pumping coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.nds">
	<code class="descname">nds</code><a class="headerlink" href="#piqs.dicke.Dicke.nds" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>int</em> -- The number of Dicke states.</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Dicke.dshape">
	<code class="descname">dshape</code><a class="headerlink" href="#piqs.dicke.Dicke.dshape" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>tuple</em> -- The shape of the Hilbert space in the Dicke or uncoupled basis.
	default: (nds, nds).</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.__repr__">
	<code class="descname">__repr__</code><span class="sig-paren">(</span><span class="sig-paren">)</span></dt>
	<dd><p>Print the current parameters of the system.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.c_ops">
	<code class="descname">c_ops</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.c_ops"><a class="headerlink" href="#piqs.dicke.Dicke.c_ops" title="Permalink to this definition">¶</a></dt>
	<dd><p>Build collapse operators in the full Hilbert space 2^N.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>c_ops_list</strong> -- The list with the collapse operators in the 2^N Hilbert space.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.coefficient_matrix">
	<code class="descname">coefficient_matrix</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.coefficient_matrix"><a class="headerlink" href="#piqs.dicke.Dicke.coefficient_matrix" title="Permalink to this definition">¶</a></dt>
	<dd><p>Build coefficient matrix for ODE for a diagonal problem.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>M</strong> -- The matrix M of the coefficients for the ODE dp/dt = M p.
	p is the vector of the diagonal matrix elements
	of the density matrix rho in the Dicke basis.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body">ndarray</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.lindbladian">
	<code class="descname">lindbladian</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.lindbladian"><a class="headerlink" href="#piqs.dicke.Dicke.lindbladian" title="Permalink to this definition">¶</a></dt>
	<dd><p>Build the Lindbladian superoperator of the dissipative dynamics.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>lindbladian</strong> -- The Lindbladian matrix as a <cite>qutip.Qobj</cite>.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.liouvillian">
	<code class="descname">liouvillian</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.liouvillian"><a class="headerlink" href="#piqs.dicke.Dicke.liouvillian" title="Permalink to this definition">¶</a></dt>
	<dd><p>Build the total Liouvillian using the Dicke basis.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Returns:</th><td class="field-body"><strong>liouv</strong> -- The Liouvillian matrix for the system.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Return type:</th><td class="field-body"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.pisolve">
	<code class="descname">pisolve</code><span class="sig-paren">(</span><em>initial_state</em>, <em>tlist</em>, <em>options=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.pisolve"><a class="headerlink" href="#piqs.dicke.Dicke.pisolve" title="Permalink to this definition">¶</a></dt>
	<dd><p>Solve for diagonal Hamiltonians and initial states faster.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>initial_state</strong> -- An initial state specified as a density matrix of <cite>qutip.Qbj</cite> type</li>
	<li><strong>tlist</strong> (<em>ndarray</em>) -- A 1D numpy array of list of timesteps to integrate</li>
	<li><strong>options</strong> -- The options for the solver.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>result</strong> -- A dictionary of the type <cite>qutip.solver.Result</cite> which holds the
	results of the evolution.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Dicke.prune_eigenstates">
	<code class="descname">prune_eigenstates</code><span class="sig-paren">(</span><em>liouvillian</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Dicke.prune_eigenstates"><a class="headerlink" href="#piqs.dicke.Dicke.prune_eigenstates" title="Permalink to this definition">¶</a></dt>
	<dd><p>Remove spurious eigenvalues and eigenvectors of the Liouvillian.</p>
	<p>Spurious means that the given eigenvector has elements outside of the
	block-diagonal matrix.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>liouvillian_eigenstates</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)"><em>list</em></a>) -- A list with the eigenvalues and eigenvectors of the Liouvillian
	including spurious ones.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>correct_eigenstates</strong> -- The list with the correct eigenvalues and eigenvectors of the
	Liouvillian.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	</dd></dl>

	<dl class="class">
	<dt id="piqs.dicke.Pim">
	<em class="property">class </em><code class="descclassname">piqs.dicke.</code><code class="descname">Pim</code><span class="sig-paren">(</span><em>N</em>, <em>emission=0.0</em>, <em>dephasing=0</em>, <em>pumping=0</em>, <em>collective_emission=0</em>, <em>collective_pumping=0</em>, <em>collective_dephasing=0</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim"><a class="headerlink" href="#piqs.dicke.Pim" title="Permalink to this definition">¶</a></dt>
	<dd><p>Bases: <a class="reference external" href="https://docs.python.org/3/library/functions.html#object" title="(in Python v3.6)"><code class="xref py py-class docutils literal notranslate"><span class="pre">object</span></code></a></p>
	<p>The Permutation Invariant Matrix class.</p>
	<p>Initialize the class with the parameters for generating a Permutation
	Invariant matrix which evolves a given diagonal initial state <cite>p</cite> as:</p>
	<blockquote>
	<div>dp/dt = Mp</div></blockquote>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent emission coefficient (also nonradiative emission).
	default: 0.0</li>
	<li><strong>dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Local dephasing coefficient.
	default: 0.0</li>
	<li><strong>pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective (superradiant) emmission coefficient.
	default: 0.0</li>
	<li><strong>collective_pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective dephasing coefficient.
	default: 0.0</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	<dl class="attribute">
	<dt id="piqs.dicke.Pim.N">
	<code class="descname">N</code><a class="headerlink" href="#piqs.dicke.Pim.N" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>int</em> -- The number of two-level systems.</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.emission">
	<code class="descname">emission</code><a class="headerlink" href="#piqs.dicke.Pim.emission" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Incoherent emission coefficient (also nonradiative emission).
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.dephasing">
	<code class="descname">dephasing</code><a class="headerlink" href="#piqs.dicke.Pim.dephasing" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Local dephasing coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.pumping">
	<code class="descname">pumping</code><a class="headerlink" href="#piqs.dicke.Pim.pumping" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Incoherent pumping coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.collective_emission">
	<code class="descname">collective_emission</code><a class="headerlink" href="#piqs.dicke.Pim.collective_emission" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective (superradiant) emmission coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.collective_dephasing">
	<code class="descname">collective_dephasing</code><a class="headerlink" href="#piqs.dicke.Pim.collective_dephasing" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective dephasing coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.collective_pumping">
	<code class="descname">collective_pumping</code><a class="headerlink" href="#piqs.dicke.Pim.collective_pumping" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>float</em> -- Collective pumping coefficient.
	default: 0.0</p>
	</dd></dl>

	<dl class="attribute">
	<dt id="piqs.dicke.Pim.M">
	<code class="descname">M</code><a class="headerlink" href="#piqs.dicke.Pim.M" title="Permalink to this definition">¶</a></dt>
	<dd><p><em>dict</em> -- A nested dictionary of the structure {row: {col: val}} which holds
	non zero elements of the matrix M</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.calculate_j_m">
	<code class="descname">calculate_j_m</code><span class="sig-paren">(</span><em>dicke_row</em>, <em>dicke_col</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.calculate_j_m"><a class="headerlink" href="#piqs.dicke.Pim.calculate_j_m" title="Permalink to this definition">¶</a></dt>
	<dd><p>Get the value of j and m for the particular Dicke space element.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>dicke_row</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Row index of the element in Dicke space which needs to be checked.</li>
	<li><strong>dicke_col</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Column index of the element in Dicke space which needs to be
	checked.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>j, m</strong> -- The j and m values.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)">float</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.calculate_k">
	<code class="descname">calculate_k</code><span class="sig-paren">(</span><em>dicke_row</em>, <em>dicke_col</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.calculate_k"><a class="headerlink" href="#piqs.dicke.Pim.calculate_k" title="Permalink to this definition">¶</a></dt>
	<dd><p>Get k value from the current row and column element in the Dicke space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>dicke_row</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Row index of the element in Dicke space which needs to be checked.</li>
	<li><strong>dicke_col</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Column index of the element in Dicke space which needs to be
	checked.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>k</strong> -- The row index for the matrix M for given Dicke space
	element</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.coefficient_matrix">
	<code class="descname">coefficient_matrix</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.coefficient_matrix"><a class="headerlink" href="#piqs.dicke.Pim.coefficient_matrix" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the matrix M governing the dynamics for diagonal cases.</p>
	<p>If the initial density matrix and the Hamiltonian is diagonal, the
	evolution of the system is given by the simple ODE: dp/dt = Mp.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.isdicke">
	<code class="descname">isdicke</code><span class="sig-paren">(</span><em>dicke_row</em>, <em>dicke_col</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.isdicke"><a class="headerlink" href="#piqs.dicke.Pim.isdicke" title="Permalink to this definition">¶</a></dt>
	<dd><p>Check if an element in a matrix is a valid element in the Dicke space.
	Dicke row: j value index. Dicke column: m value index.
	The function returns True if the element exists in the Dicke space and
	False otherwise.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
	<li><strong>dicke_row</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Row index of the element in Dicke space which needs to be checked.</li>
	<li><strong>dicke_col</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Column index of the element in Dicke space which needs to be
	checked.</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.solve">
	<code class="descname">solve</code><span class="sig-paren">(</span><em>rho0</em>, <em>tlist</em>, <em>options=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.solve"><a class="headerlink" href="#piqs.dicke.Pim.solve" title="Permalink to this definition">¶</a></dt>
	<dd><p>Solve the ODE for the evolution of diagonal states and Hamiltonians.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau1">
	<code class="descname">tau1</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau1"><a class="headerlink" href="#piqs.dicke.Pim.tau1" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_jmm.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau2">
	<code class="descname">tau2</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau2"><a class="headerlink" href="#piqs.dicke.Pim.tau2" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_jm+1m+1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau3">
	<code class="descname">tau3</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau3"><a class="headerlink" href="#piqs.dicke.Pim.tau3" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j+1m+1m+1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau4">
	<code class="descname">tau4</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau4"><a class="headerlink" href="#piqs.dicke.Pim.tau4" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j-1m+1m+1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau5">
	<code class="descname">tau5</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau5"><a class="headerlink" href="#piqs.dicke.Pim.tau5" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j+1mm.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau6">
	<code class="descname">tau6</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau6"><a class="headerlink" href="#piqs.dicke.Pim.tau6" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j-1mm.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau7">
	<code class="descname">tau7</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau7"><a class="headerlink" href="#piqs.dicke.Pim.tau7" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j+1m-1m-1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau8">
	<code class="descname">tau8</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau8"><a class="headerlink" href="#piqs.dicke.Pim.tau8" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_jm-1m-1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau9">
	<code class="descname">tau9</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau9"><a class="headerlink" href="#piqs.dicke.Pim.tau9" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the element of the coefficient matrix relative to p_j-1m-1m-1.</p>
	</dd></dl>

	<dl class="method">
	<dt id="piqs.dicke.Pim.tau_valid">
	<code class="descname">tau_valid</code><span class="sig-paren">(</span><em>dicke_row</em>, <em>dicke_col</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#Pim.tau_valid"><a class="headerlink" href="#piqs.dicke.Pim.tau_valid" title="Permalink to this definition">¶</a></dt>
	<dd><p>Find the Tau functions which are valid for this value of (dicke_row,
	dicke_col) given the number of TLS. This calculates the valid tau
	values and reurns a dictionary specifying the tau function name and
	the value.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>dicke_row</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Row index of the element in Dicke space which needs to be checked.</li>
	<li><strong>dicke_col</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Column index of the element in Dicke space which needs to be
	checked.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>taus</strong> -- A dictionary of key, val as {tau: value} consisting of the valid
	taus for this row and column of the Dicke space element.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#dict" title="(in Python v3.6)">dict</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.am">
	<code class="descclassname">piqs.dicke.</code><code class="descname">am</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#am"><a class="headerlink" href="#piqs.dicke.am" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the coefficient <cite>am</cite> by applying <a href="#id7"><span class="problematic" id="id8">J_</span></a>- <a href="#id1"><span class="problematic" id="id2">|</span></a>j, m&gt;.</p>
	<p>The action of am is given by:
	<span class="math notranslate nohighlight">\(J_{-}|j, m\rangle = A_{-}(j, m)|j, m-1\rangle\)</span></p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>j</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The value for j.</li>
	<li><strong>m</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The value for m.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>a_minus</strong> -- The value of <span class="math notranslate nohighlight">\(a_{-}\)</span>.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)">float</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.ap">
	<code class="descclassname">piqs.dicke.</code><code class="descname">ap</code><span class="sig-paren">(</span><em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#ap"><a class="headerlink" href="#piqs.dicke.ap" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the coefficient <cite>ap</cite> by applying J_+ <a href="#id3"><span class="problematic" id="id4">|</span></a>j, m&gt;.</p>
	<p>The action of ap is given by:
	<span class="math notranslate nohighlight">\(J_{+}|j, m\rangle = A_{+}(j, m)|j, m+1\rangle\)</span></p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>j</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The value for j.</li>
	<li><strong>m</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The value for m.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>a_plus</strong> -- The value of <span class="math notranslate nohighlight">\(a_{+}\)</span>.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)">float</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.block_matrix">
	<code class="descclassname">piqs.dicke.</code><code class="descname">block_matrix</code><span class="sig-paren">(</span><em>N</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#block_matrix"><a class="headerlink" href="#piqs.dicke.block_matrix" title="Permalink to this definition">¶</a></dt>
	<dd><p>Construct the block-diagonal matrix for the Dicke basis.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Number of two-level systems.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>block_matr</strong> -- A 2D block-diagonal matrix of ones with dimension (nds,nds),
	where nds is the number of Dicke states for N two-level
	systems.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body">ndarray</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.collapse_uncoupled">
	<code class="descclassname">piqs.dicke.</code><code class="descname">collapse_uncoupled</code><span class="sig-paren">(</span><em>N</em>, <em>emission=0.0</em>, <em>dephasing=0.0</em>, <em>pumping=0.0</em>, <em>collective_emission=0.0</em>, <em>collective_dephasing=0.0</em>, <em>collective_pumping=0.0</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#collapse_uncoupled"><a class="headerlink" href="#piqs.dicke.collapse_uncoupled" title="Permalink to this definition">¶</a></dt>
	<dd><p>Create the collapse operators (c_ops) of the Lindbladian in the uncoupled basis.</p>
	<p>These operators are in the uncoupled basis of the two-level system
	(TLS) SU(2) Pauli matrices.</p>
	<p class="rubric">Notes</p>
	<p>The collapse operator list can be given to <cite>qutip.mesolve</cite>.
	Notice that the operators are placed in a Hilbert space of dimension
	<span class="math notranslate nohighlight">\(2^N\)</span>
	Thus the method is suitable only for small N (of the order of 10).</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent emission coefficient (also nonradiative emission).
	default: 0.0</li>
	<li><strong>dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Local dephasing coefficient.
	default: 0.0</li>
	<li><strong>pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Incoherent pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_emission</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective (superradiant) emmission coefficient.
	default: 0.0</li>
	<li><strong>collective_pumping</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective pumping coefficient.
	default: 0.0</li>
	<li><strong>collective_dephasing</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Collective dephasing coefficient.
	default: 0.0</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>c_ops</strong> -- The list of collapse operators as <cite>qutip.Qobj</cite> for the system.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.css">
	<code class="descclassname">piqs.dicke.</code><code class="descname">css</code><span class="sig-paren">(</span><em>N</em>, <em>x=0.7071067811865475</em>, <em>y=0.7071067811865475</em>, <em>basis='dicke'</em>, <em>coordinates='cartesian'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#css"><a class="headerlink" href="#piqs.dicke.css" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the density matrix of the Coherent Spin State (CSS).</p>
	<p>It can be defined as,
	<span class="math notranslate nohighlight">\(|CSS \rangle = \prod_i^N(a|1\rangle_i + b|0\rangle_i)\)</span>
	with <span class="math notranslate nohighlight">\(a = sin(\frac{\theta}{2})\)</span>,
	<span class="math notranslate nohighlight">\(b = e^{i \phi}\cos(\frac{\theta}{2})\)</span>.
	The default basis is that of Dicke space
	<span class="math notranslate nohighlight">\(|j, m\rangle \langle j, m'|\)</span>.
	The default state is the symmetric CSS,
	<span class="math notranslate nohighlight">\(|CSS\rangle = |+\rangle\)</span>.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>y</strong> (<em>x</em><em>,</em>) -- The coefficients of the CSS state.</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis to use. Either &quot;dicke&quot; or &quot;uncoupled&quot;.</li>
	<li><strong>coordinates</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- Either &quot;cartesian&quot; or &quot;polar&quot;. If polar then the coefficients
	are constructed as sin(x/2), cos(x/2)e^(iy).</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>rho</strong> -- The CSS state density matrix.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.dicke">
	<code class="descclassname">piqs.dicke.</code><code class="descname">dicke</code><span class="sig-paren">(</span><em>N</em>, <em>j</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#dicke"><a class="headerlink" href="#piqs.dicke.dicke" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate a Dicke state as a pure density matrix in the Dicke basis.</p>
	<p>For instance, the superradiant state given by
	<span class="math notranslate nohighlight">\(|j, m\rangle = |1, 0\rangle\)</span> for N = 2,
	and the state is represented as a density matrix of size (nds, nds) or
	(4, 4), with the (1, 1) element set to 1.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>j</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The eigenvalue j of the Dicke state (j, m).</li>
	<li><strong>m</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The eigenvalue m of the Dicke state (j, m).</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>rho</strong> -- The density matrix.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.dicke_basis">
	<code class="descclassname">piqs.dicke.</code><code class="descname">dicke_basis</code><span class="sig-paren">(</span><em>N</em>, <em>jmm1=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#dicke_basis"><a class="headerlink" href="#piqs.dicke.dicke_basis" title="Permalink to this definition">¶</a></dt>
	<dd><p>Initialize the density matrix of a Dicke state for several (j, m, m1).</p>
	<p>This function can be used to build arbitrary states in the Dicke basis
	<span class="math notranslate nohighlight">\(|j, m\rangle \langle j, m^{\prime}|\)</span>. We create coefficients for each
	(j, m, m1) value in the dictionary jmm1. The mapping for the (i, k)
	index of the density matrix to the <a href="#id5"><span class="problematic" id="id6">|</span></a>j, m&gt; values is given by the
	cythonized function <cite>jmm1_dictionary</cite>. A density matrix is created from
	the given dictionary of coefficients for each (j, m, m1).</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>jmm1</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#dict" title="(in Python v3.6)"><em>dict</em></a>) -- A dictionary of {(j, m, m1): p} that gives a density p for the
	(j, m, m1) matrix element.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>rho</strong> -- The density matrix in the Dicke basis.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.energy_degeneracy">
	<code class="descclassname">piqs.dicke.</code><code class="descname">energy_degeneracy</code><span class="sig-paren">(</span><em>N</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#energy_degeneracy"><a class="headerlink" href="#piqs.dicke.energy_degeneracy" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the number of Dicke states with same energy.</p>
	<p>The use of the <cite>Decimals</cite> class allows to explore N &gt; 1000,
	unlike the built-in function <cite>scipy.special.binom</cite></p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>m</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Total spin z-axis projection eigenvalue.
	This is proportional to the total energy.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>degeneracy</strong> -- The energy degeneracy</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.excited">
	<code class="descclassname">piqs.dicke.</code><code class="descname">excited</code><span class="sig-paren">(</span><em>N</em>, <em>basis='dicke'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#excited"><a class="headerlink" href="#piqs.dicke.excited" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the density matrix for the excited state.</p>
	<p>This state is given by (N/2, N/2) in the default Dicke basis. If the
	argument <cite>basis</cite> is &quot;uncoupled&quot; then it generates the state in a
	<span class="math notranslate nohighlight">\(2^N\)</span> dim Hilbert space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis to use. Either &quot;dicke&quot; or &quot;uncoupled&quot;.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>state</strong> -- The excited state density matrix in the requested basis.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.ghz">
	<code class="descclassname">piqs.dicke.</code><code class="descname">ghz</code><span class="sig-paren">(</span><em>N</em>, <em>basis='dicke'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#ghz"><a class="headerlink" href="#piqs.dicke.ghz" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the density matrix of the GHZ state.</p>
	<p>If the argument <cite>basis</cite> is &quot;uncoupled&quot; then it generates the state
	in a <span class="math notranslate nohighlight">\(2^N\)</span> dim Hilbert space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis to use. Either &quot;dicke&quot; or &quot;uncoupled&quot;.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>state</strong> -- The GHZ state density matrix in the requested basis.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.ground">
	<code class="descclassname">piqs.dicke.</code><code class="descname">ground</code><span class="sig-paren">(</span><em>N</em>, <em>basis='dicke'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#ground"><a class="headerlink" href="#piqs.dicke.ground" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the density matrix of the ground state.</p>
	<p>This state is given by (N/2, -N/2) in the Dicke basis. If the argument
	<cite>basis</cite> is &quot;uncoupled&quot; then it generates the state in a
	<span class="math notranslate nohighlight">\(2^N\)</span> dim Hilbert space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis to use. Either &quot;dicke&quot; or &quot;uncoupled&quot;</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>state</strong> -- The ground state density matrix in the requested basis.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.identity_uncoupled">
	<code class="descclassname">piqs.dicke.</code><code class="descname">identity_uncoupled</code><span class="sig-paren">(</span><em>N</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#identity_uncoupled"><a class="headerlink" href="#piqs.dicke.identity_uncoupled" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the identity in a <span class="math notranslate nohighlight">\(2^N\)</span> dimensional Hilbert space.</p>
	<p>The identity matrix is formed from the tensor product of N TLSs.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>identity</strong> -- The identity matrix.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.isdiagonal">
	<code class="descclassname">piqs.dicke.</code><code class="descname">isdiagonal</code><span class="sig-paren">(</span><em>mat</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#isdiagonal"><a class="headerlink" href="#piqs.dicke.isdiagonal" title="Permalink to this definition">¶</a></dt>
	<dd><p>Check if the input matrix is diagonal.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>mat</strong> (<em>ndarray/Qobj</em>) -- A 2D numpy array</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>diag</strong> -- True/False depending on whether the input matrix is diagonal.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/functions.html#bool" title="(in Python v3.6)">bool</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.jspin">
	<code class="descclassname">piqs.dicke.</code><code class="descname">jspin</code><span class="sig-paren">(</span><em>N</em>, <em>op=None</em>, <em>basis='dicke'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#jspin"><a class="headerlink" href="#piqs.dicke.jspin" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the list of collective operators of the total algebra.</p>
	<p>The Dicke basis <span class="math notranslate nohighlight">\(|j,m\rangle\langle j,m'|\)</span> is used by
	default. Otherwise with &quot;uncoupled&quot; the operators are in a
	<span class="math notranslate nohighlight">\(2^N\)</span> space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- Number of two-level systems.</li>
	<li><strong>op</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The operator to return 'x','y','z','+','-'.
	If no operator given, then output is the list of operators
	for ['x','y','z'].</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis of the operators - &quot;dicke&quot; or &quot;uncoupled&quot;
	default: &quot;dicke&quot;.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>j_alg</strong> -- A list of <cite>qutip.Qobj</cite> representing all the operators in
	the &quot;dicke&quot; or &quot;uncoupled&quot; basis or a single operator requested.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a> or :class: qutip.Qobj</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.m_degeneracy">
	<code class="descclassname">piqs.dicke.</code><code class="descname">m_degeneracy</code><span class="sig-paren">(</span><em>N</em>, <em>m</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#m_degeneracy"><a class="headerlink" href="#piqs.dicke.m_degeneracy" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the number of Dicke states <span class="math notranslate nohighlight">\(|j, m\rangle\)</span> with
	same energy.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>m</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Total spin z-axis projection eigenvalue (proportional to the total
	energy).</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>degeneracy</strong> -- The m-degeneracy.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.num_dicke_ladders">
	<code class="descclassname">piqs.dicke.</code><code class="descname">num_dicke_ladders</code><span class="sig-paren">(</span><em>N</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#num_dicke_ladders"><a class="headerlink" href="#piqs.dicke.num_dicke_ladders" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the total number of ladders in the Dicke space.</p>
	<p>For a collection of N two-level systems it counts how many different
	&quot;j&quot; exist or the number of blocks in the block-diagonal matrix.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>Nj</strong> -- The number of Dicke ladders.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.num_dicke_states">
	<code class="descclassname">piqs.dicke.</code><code class="descname">num_dicke_states</code><span class="sig-paren">(</span><em>N</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#num_dicke_states"><a class="headerlink" href="#piqs.dicke.num_dicke_states" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the number of Dicke states.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>nds</strong> -- The number of Dicke states.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.num_tls">
	<code class="descclassname">piqs.dicke.</code><code class="descname">num_tls</code><span class="sig-paren">(</span><em>nds</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#num_tls"><a class="headerlink" href="#piqs.dicke.num_tls" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the number of two-level systems.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>nds</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of Dicke states.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>N</strong> -- The number of two-level systems.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.spin_algebra">
	<code class="descclassname">piqs.dicke.</code><code class="descname">spin_algebra</code><span class="sig-paren">(</span><em>N</em>, <em>op=None</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#spin_algebra"><a class="headerlink" href="#piqs.dicke.spin_algebra" title="Permalink to this definition">¶</a></dt>
	<dd><p>Create the list [sx, sy, sz] with the spin operators.</p>
	<p>The operators are constructed for a collection of N two-level systems
	(TLSs). Each element of the list, i.e., sx, is a vector of <cite>qutip.Qobj</cite>
	objects (spin matrices), as it cointains the list of the SU(2) Pauli
	matrices for the N TLSs. Each TLS operator sx[i], with i = 0, ..., (N-1),
	is placed in a <span class="math notranslate nohighlight">\(2^N\)</span>-dimensional Hilbert space.</p>
	<p class="rubric">Notes</p>
	<p>sx[i] is <span class="math notranslate nohighlight">\(\frac{\sigma_x}{2}\)</span> in the composite Hilbert space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><strong>spin_operators</strong> -- A list of <cite>qutip.Qobj</cite> operators - [sx, sy, sz] or the
	requested operator.</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#list" title="(in Python v3.6)">list</a> or :class: qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.state_degeneracy">
	<code class="descclassname">piqs.dicke.</code><code class="descname">state_degeneracy</code><span class="sig-paren">(</span><em>N</em>, <em>j</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#state_degeneracy"><a class="headerlink" href="#piqs.dicke.state_degeneracy" title="Permalink to this definition">¶</a></dt>
	<dd><p>Calculate the degeneracy of the Dicke state.</p>
	<p>Each state <span class="math notranslate nohighlight">\(|j, m\rangle\)</span> includes D(N,j) irreducible
	representations <span class="math notranslate nohighlight">\(|j, m, \alpha\rangle\)</span>.</p>
	<p>Uses Decimals to calculate higher numerator and denominators numbers.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>j</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- Total spin eigenvalue (cooperativity).</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>degeneracy</strong> -- The state degeneracy.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)">int</a></p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.superradiant">
	<code class="descclassname">piqs.dicke.</code><code class="descname">superradiant</code><span class="sig-paren">(</span><em>N</em>, <em>basis='dicke'</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#superradiant"><a class="headerlink" href="#piqs.dicke.superradiant" title="Permalink to this definition">¶</a></dt>
	<dd><p>Generate the density matrix of the superradiant state.</p>
	<p>This state is given by (N/2, 0) or (N/2, 0.5) in the Dicke basis.
	If the argument <cite>basis</cite> is &quot;uncoupled&quot; then it generates the state
	in a 2**N dim Hilbert space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first simple">
	<li><strong>N</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The number of two-level systems.</li>
	<li><strong>basis</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The basis to use. Either &quot;dicke&quot; or &quot;uncoupled&quot;.</li>
	</ul>
	</td>
	</tr>
	<tr class="field-even field"><th class="field-name">Returns:</th><td class="field-body"><p class="first"><strong>state</strong> -- The superradiant state density matrix in the requested basis.</p>
	</td>
	</tr>
	<tr class="field-odd field"><th class="field-name">Return type:</th><td class="field-body"><p class="first last"><table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">class:</th><td class="field-body">qutip.Qobj</td>
	</tr>
	</tbody>
	</table>
	</p>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	<dl class="function">
	<dt id="piqs.dicke.tau_column">
	<code class="descclassname">piqs.dicke.</code><code class="descname">tau_column</code><span class="sig-paren">(</span><em>tau</em>, <em>k</em>, <em>j</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/dicke.html#tau_column"><a class="headerlink" href="#piqs.dicke.tau_column" title="Permalink to this definition">¶</a></dt>
	<dd><p>Determine the column index for the non-zero elements of the matrix for a
	particular row <cite>k</cite> and the value of <cite>j</cite> from the Dicke space.</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><ul class="first last simple">
	<li><strong>tau</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The tau function to check for this <cite>k</cite> and <cite>j</cite>.</li>
	<li><strong>k</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#int" title="(in Python v3.6)"><em>int</em></a>) -- The row of the matrix M for which the non zero elements have
	to be calculated.</li>
	<li><strong>j</strong> (<a class="reference external" href="https://docs.python.org/3/library/functions.html#float" title="(in Python v3.6)"><em>float</em></a>) -- The value of <cite>j</cite> for this row.</li>
	</ul>
	</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	</div>
	<div class="section" id="module-piqs.cite">
	<span id="citation-generator"></span><h2>Citation generator<a class="headerlink" href="#module-piqs.cite" title="Permalink to this headline">¶</a></h2>
	<p>Citation generator for PIQS</p>
	<dl class="function">
	<dt id="piqs.cite.cite">
	<code class="descclassname">piqs.cite.</code><code class="descname">cite</code><span class="sig-paren">(</span><em>path=None</em>, <em>verbose=True</em><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/cite.html#cite"><a class="headerlink" href="#piqs.cite.cite" title="Permalink to this definition">¶</a></dt>
	<dd><p>Citation information and bibtex generator for PIQS</p>
	<table class="docutils field-list" frame="void" rules="none">
	<col class="field-name" />
	<col class="field-body" />
	<tbody valign="top">
	<tr class="field-odd field"><th class="field-name">Parameters:</th><td class="field-body"><strong>path</strong> (<a class="reference external" href="https://docs.python.org/3/library/stdtypes.html#str" title="(in Python v3.6)"><em>str</em></a>) -- The complete directory path to generate the bibtex file.
	If not specified then the citation will be generated in cwd</td>
	</tr>
	</tbody>
	</table>
	</dd></dl>

	</div>
	<div class="section" id="module-piqs.about">
	<span id="about"></span><h2>About<a class="headerlink" href="#module-piqs.about" title="Permalink to this headline">¶</a></h2>
	<p>Command line output of information on QuTiP and dependencies.</p>
	<dl class="function">
	<dt id="piqs.about.about">
	<code class="descclassname">piqs.about.</code><code class="descname">about</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="reference internal" href="_modules/piqs/about.html#about"><a class="headerlink" href="#piqs.about.about" title="Permalink to this definition">¶</a></dt>
	<dd><p>About box for PIQS.</p>
	</dd></dl>

	</div>
	</div>


	           </div>
	           
	          </div>
	          <footer>
	  
	    <div class="rst-footer-buttons" role="navigation" aria-label="footer navigation">
	      
	        <a href="developers.html" class="btn btn-neutral float-right" title="Developers" accesskey="n" rel="next">Next <span class="fa fa-arrow-circle-right"></span></a>
	      
	      
	        <a href="examples/spin_squeezing.html" class="btn btn-neutral" title="Spin Squeezing" accesskey="p" rel="prev"><span class="fa fa-arrow-circle-left"></span> Previous</a>
	      
	    </div>
	  

	  <hr/>

	  <div role="contentinfo">
	    <p>
	        &copy; Copyright 2018, Nathan Shammah, Shahnawaz Ahmed.

	    </p>
	  </div>
	  Built with <a href="http://sphinx-doc.org/">Sphinx</a> using a <a href="https://github.com/rtfd/sphinx_rtd_theme">theme</a> provided by <a href="https://readthedocs.org">Read the Docs</a>. 

	</footer>

	        </div>
	      </div>

	    </section>

	  </div>
	  


	  

	    <script type="text/javascript">
	        var DOCUMENTATION_OPTIONS = {
	            URL_ROOT:'./',
	            VERSION:'1.1-dev',
	            LANGUAGE:'python',
	            COLLAPSE_INDEX:false,
	            FILE_SUFFIX:'.html',
	            HAS_SOURCE:  true,
	            SOURCELINK_SUFFIX: '.txt'
	        };
	    </script>
	      <script type="text/javascript" src="_static/jquery.js"></script>
	      <script type="text/javascript" src="_static/underscore.js"></script>
	      <script type="text/javascript" src="_static/doctools.js"></script>
	      <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

	  

	  
	  
	    <script type="text/javascript" src="_static/js/theme.js"></script>
	  

	  <script type="text/javascript">
	      jQuery(function () {
	          SphinxRtdTheme.Navigation.enable(true);
	      });
	  </script> 
"""The generated feature matrix: feature x adapter x way x mode x selection (``make matrix``).

* ``model.py`` — the dimensions (ADAPTERS, WAYS, MODES), a row (``Feature``) and what a cell may
  be instead of a plain test: ``NA`` (with why), ``Gap`` (an audit gap or plan item: strict
  xfail), ``Bug`` (a real failure this matrix found: xfail on its own exception).
* ``features.py`` — the FEATURES table and each feature's scenario; ``way2.py`` the Way 2
  scenarios (the blocks without a Harness); ``extensions.py`` the features in flight.
* ``dimensions.py`` — the SELECTION dimension: the switches, the pending ones (``without=``),
  and the selections (all, none, each alone on, each alone off, all-pairs rows: ``allpairs.py``).
* ``world.py`` — one cell's deployment (the switched blocks, over the suite's fakes), the agent
  under test (``tests.support.adapters.BUILDERS``), the run driven in the cell's mode, and
  ``World.verify``: off leaves no trace, on did its part, events are well formed.
* ``generate.py`` — every cell, and ``KNOWN``: the real failures and where they hold.
* ``report.py`` — ``python -m tests.matrix.report build/matrix-*.xml`` writes the Markdown
  matrix from the junit results (what ran, not what the tables say).

Extending it
------------

* **A feature.** Write ``async def scenario(w: World)``: build tools with ``kit.Desk``, run
  ``await w.go(tools, plan)`` (a plan of ``(tool, args)`` calls the scripted model makes; the
  answer is ``"Done. <last result>"``), and assert the behaviour when ``w.on`` and no trace when
  not. Add a ``Feature`` to ``FEATURES`` with its audit id, how it is turned on (``how``), the
  switches it needs (``needs``), and the exceptions: ``adapters``/``modes``/``ways``/``cells``
  mapped to ``NA(reason)`` or ``Gap(id, why)``; ``way2`` is its Way 2 scenario or a note.
* **A feature that lands** (an extension point or a ``Gap``): its cells XPASS and fail the
  suite; remove the ``Gap`` (and turn an ``extensions.py`` probe into a real scenario).
* **A switch** (``without=`` landing, hooks...): move it from ``dimensions.PENDING`` to
  ``SWITCHES`` (with ``requires``), give it its effect in ``World._made``/``World.agent``,
  and its on/off checks in ``World.verify``. Every selection row picks it up, the all-pairs
  rows included.
* **An adapter, way or mode**: add it to ``model.py``; a mode needs its driver in
  ``World._start``/``_resume``/``_settled`` (and ``World.cancel``).
* **A bug fixed**: remove its ``KNOWN`` entry in ``generate.py``.

``MATRIX_SHARD=i/n`` runs one of n shards (``conftest.py``); ``make matrix`` runs
``MATRIX_SHARDS`` of them at once and writes ``build/matrix.md``.
"""

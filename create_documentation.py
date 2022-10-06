import pydoc
import langevin_harmonic_osc_simulator

with open("doc.txt", "w") as f:
  print(pydoc.render_doc(langevin_harmonic_osc_simulator, renderer=pydoc.plaintext), file=f)

with open("doc.html", "w") as f:
  print(pydoc.render_doc(langevin_harmonic_osc_simulator, renderer=pydoc.html), file=f)


 
# Evaluation Rubric

Use this rubric when comparing current-round candidates against the parent molecule.

## Required dimensions
- Goal fit: does the candidate move the target endpoint or property in the correct direction?
- Parent continuity: does it preserve essential scaffold and pharmacophore context?
- Medchem risk: does it introduce toxicophore, instability, or synthesis concerns?
- Evidence quality: are the claims supported by direct tool outputs, case evidence, or established medicinal chemistry reasoning?

## Ranking behavior
- A candidate with modest numerical improvement but clean medchem profile can outrank a numerically stronger but risky analog.
- If endpoint support is missing, say so explicitly and rank mainly on medicinal chemistry judgment.
- Keep all candidates in the final output, even low-ranked ones.

## Common penalties
- Large unjustified edit distance from the parent molecule.
- Structural alerts or obvious instability motifs.
- Loss of likely key binding or recognition features.
- Claims that are not supported by the actual structural modification.

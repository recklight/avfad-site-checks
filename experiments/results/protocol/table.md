| detector | material | HSA UAR | HSA AUC | HSA AUC inside strata | everyone UAR | everyone AUC | non-phonation, everyone -> HSA |
|---|---|---|---|---|---|---|---|
| handcrafted | /a/ | 0.553 | 0.580 | 0.573 [0.519, 0.626] | 0.820 | 0.872 | 0.814 -> 0.474 |
| HuBERT (frozen) | /a/ | 0.621 | 0.666 | 0.656 [0.602, 0.712] | 0.835 | 0.908 | - |
| handcrafted | six sentences | 0.477 | 0.495 | 0.486 [0.437, 0.534] | 0.820 | 0.863 | 0.835 -> 0.512 |
| handcrafted | reading | 0.541 | 0.565 | 0.560 [0.509, 0.605] | 0.830 | 0.882 | 0.826 -> 0.491 |
| HuBERT (frozen) | six sentences | 0.600 | 0.626 | 0.610 [0.566, 0.655] | 0.859 | 0.912 | - |
| HuBERT (frozen) | reading | 0.609 | 0.644 | 0.622 [0.568, 0.671] | 0.861 | 0.917 | - |
| ComParE + SVM (their setting) | /a/ | 0.504 | 0.641 | 0.633 [0.574, 0.690] | 0.858 | 0.910 | - |
| ComParE + SVM (their setting) | six sentences | not run | not run | not run | not run | not run | - |
| ComParE + SVM, class weight | /a/ | not run | not run | not run | not run | not run | - |
| ComParE + SVM, class weight | six sentences | not run | not run | not run | not run | not run | - |
| HuBERT, last 2 layers fine-tuned | /a/ | 0.597 | 0.651 | 0.647 [0.593, 0.703] | 0.828 | 0.914 | - |

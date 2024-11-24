# Implicit-Attention
Implementation of Implicit Deep Learning for Attention Mechanisms

## Project Structure

Below is the projects' repository structure:

```plaintext
Project 
├─ 📂utils                      
│   ├─ 📃data.py           
│   ├─ 📃model.py              
│   ├─ 📃utils.py                                     
│   ├─ 📃train.py
│ 
├─ 📃main.py
│
└─ 💲scripts                    
```

1. `main.py`: Manages the training and evaluation of both explicit and implicit models across various datasets.
2. `utils`:
   - `data.py`: Return train and validation data for each dataset (`tinystories`, `wikitext`, `tinyshakespeare` for NLP and `cifar-10` for CV).
   - `load_model.py`: Return model structures of both explicit and implicit (`GPT-2` for NLP and `ViT` for CV).
   - `utils.py`: Basic functions such as: Estimate the loss; Set global seed; Generate output text of implicit moded ...
   - `train.py`: Trains both explicit and implicit models.
3. `scripts`: Scripts to reproduce the results of our group.

## TODO

- [x] Code structure Phase 1
- [x] Docstring all functions Phase 1
- [x] Reorganize the code following the structure Phase 1
- [ ] Update the code for Computer Vision task and Vision Transformer architecture
- [ ] Code structure Phase 2
- [ ] Docstring all functions Phase 2
- [ ] Reorganize the code following the structure Phase 2
- [ ] Build efficient saving results and plots
- [ ] Inference to test and duplicate report results

This repository implements a research pipeline testing whether predictive-coding neural networks produce internal representations that resemble human EEG activity better than conventional supervised networks.

## Overall pipeline

```text
Image pairs
   ↓
Train recurrent networks under 7 learning objectives
   ↓
Extract hidden-layer activations over 15 time steps
   ↓
Convert model activations into RDMs
   ↓
Convert human EEG recordings into matching RDMs
   ↓
Correlate model and EEG representational geometry
   ↓
Bayesian and residual-variance analyses
```

An RDM—representational dissimilarity matrix—describes how differently a system represents each pair of experimental conditions. Six conditions produce 15 unique pairwise distances.

## 1. Experimental data

The network sees pairs consisting of a leading scene followed by a trailing scene. The transition probabilities are defined in [load_dataset.py](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/load_dataset.py:9).

Examples:

- Barn or beach → church with probability 0.75: valid/predictable.
- Barn or beach → conference room with probability 0.25: invalid/surprising.
- Library or restaurant use the opposite probabilities.
- Cave → castle or forest with probability 0.5: neutral controls.

Images are resized to `40 × 40`, normalized, augmented, flattened, and returned in pairs by [create_paired_dataset](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/load_dataset.py:104).

The optional [run_generate_data.py](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_generate_data.py) uses Stable Diffusion 2 to generate 100 additional photographs per category.

## 2. Temporal stimulus sequence

[run_train_models.py](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_train_models.py:53) converts each image pair into a 15-step sequence:

- Step 0: gray interval.
- Step 1: leading image.
- Steps 2–9: gray inter-stimulus interval.
- Step 10: trailing image.
- Steps 11–14: gray interval.

Thus, the recurrent network processes a sequence resembling the timing of the human EEG experiment.

## 3. Models trained

All models share two recurrent layers:

```python
SimpleRNN(256, return_sequences=True)
SimpleRNN(128, return_sequences=True)
```

The supervised models additionally have a nine-class output layer. Model construction is in [create_model](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_train_models.py:120).

Seven conditions are compared:

| Condition | What it learns |
|---|---|
| Predictive global | Both recurrent layers jointly predict future sensory input |
| Predictive local | Each recurrent layer is trained separately using a local predictive objective |
| Contrastive global | Both layers jointly distinguish likely and unlikely transitions |
| Contrastive local | Each layer is trained separately using the contrastive objective |
| Supervised | Classifies the nine image categories |
| Supervised shuffled | Same classifier, but trained using shuffled labels |
| Untrained | Randomly initialized baseline |

“Global” means ordinary end-to-end optimization. “Local” uses [model_fit_layers](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/training.py:33), which trains one recurrent layer at a time and then passes its representation to the next layer.

Training is configured for 100 epochs with Adam and checkpoints after each epoch.

### Predictive objective

[PredPriorLoss](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/training.py:156):

1. Projects a hidden state through the recurrent weights.
2. approximately inverts the recurrent layers back into pixel space.
3. compares the reconstructed prediction at one time step with the next input.

The actual implementation compares the reconstruction against the negative next input, apparently reflecting the authors’ prediction-error sign convention.

### Contrastive objective

[FFLossWithThreshold](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/training.py:186) implements a Forward-Forward-style loss. The squared activity of a representation is treated as its “goodness,” and transition probability determines the binary training target.

## 4. Model RDMs

[run_save_model_rdms.ipynb](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_save_model_rdms.ipynb) performs the following:

1. Loads each trained model.
2. Groups examples into six EEG-compatible conditions:
   `valid_church`, `valid_confroom`, `invalid_confroom`,
   `invalid_church`, `control_castle`, and `control_forest`.
3. Averages recurrent activations within each condition.
4. Computes a six-condition RDM at every model time step.
5. Saves separate RDMs for layer 1, layer 2, and the concatenation of both layers.

Although output filenames contain `spearman`, the within-model distance is actually correlation distance, `1 − Pearson correlation`. Spearman correlation is subsequently used to compare model and EEG RDM vectors.

## 5. EEG processing

[read_custom_epochs](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/load_eeg.py:12) converts custom MATLAB EEG files into MNE epoch objects. It:

- Converts microvolts to volts.
- reconstructs sampling frequency and event labels;
- loads electrode locations;
- creates an MNE montage.

[run_save_eeg_rdms.py](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_save_eeg_rdms.py:22) then:

1. Resamples EEG to 100 Hz.
2. Creates full, half-session, and quarter-session splits.
3. repairs incorrect leading/trailing alternation;
4. joins each leading epoch to its trailing epoch;
5. maps trials to the same six conditions used for models;
6. averages trials within condition;
7. computes a correlation-distance RDM at every EEG time point.

This makes the model and human data directly comparable: both contain the same 15 pairwise dissimilarities.

## 6. Main analysis

The utility functions in [analysis.py](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/src/analysis.py:7) load RDMs and correlate every model-step RDM with every subject/time EEG RDM.

[run_analysis_final.ipynb](C:/Users/khadidja/Desktop/PFE/representations_of_predictive_coding_networks/run_analysis_final.ipynb) then:

- Keeps EEG data from `0.6 s` onward, following trailing-image onset.
- averages correlations across model steps and EEG time;
- fits Bayesian hierarchical models with subject-specific intercepts;
- calculates each training condition’s posterior probability of being the best model.

The saved outputs report:

- First half of the experiment: supervised model has `P(best) = 0.936`.
- Second half: predictive-local model has `P(best) = 0.776`.

This suggests a change over the experiment: supervised representations fit early EEG responses best, whereas locally trained predictive representations fit later responses best.

However, the direct Predictive-local × Second-half interaction was uncertain:

- Mean: `0.0325`
- 95% HDI: `[-0.0254, 0.0906]`
- Directional posterior probability: `0.865`

Because the interval crosses zero, this interaction is suggestive rather than conclusive. The notebook calls posterior odds an “approximate Bayes factor”; that is not the same as a conventional marginal-likelihood Bayes factor.

## 7. Follow-up residual analysis

The notebook then asks whether predictive-local representations explain EEG structure that cannot be explained by:

- the supervised model;
- the shuffled-label model;
- the untrained model.

It uses non-negative least squares to regress those baseline RDMs out of the predictive-local RDM. The remaining residual is correlated with EEG at every combination of model step and EEG time, followed by across-subject tests and FDR correction.

Finally, the residual is compared with three theoretical RDM templates:

- Stimulus-driven structure.
- Prediction-driven structure.
- Validity-driven structure.

Shared variance is calculated as:

```text
R²(model) + R²(template) − R²(model + template)
```

The saved notebook output shows the clearest overlap for the prediction-driven template:

- Prediction-driven shared variance: approximately `0.018`
- 95% HDI: `[0.015, 0.021]`
- Posterior probability above zero: `1.000`

The stimulus-driven template was near zero. This is the analysis supporting the title’s claim that predictive networks capture prediction-related neural structure absent from ordinary supervised networks.

## Practical limitations

The repository is not self-contained:

- Dataset, EEG, model, and result paths are hard-coded under `/scratch/guetlid95/...`.
- Raw recordings, stimuli, trained models, and RDM files are not included.
- The README mentions script names without the repository’s `run_` prefixes.
- There are some encoding artifacts such as `RÂ²` and `Ã—`.
- Several plotting calls generate deprecation warnings.
- Some statistical labels are potentially misleading—for example, a credibly negative learning-stage effect is described as “evidence for null” because the helper only tests the positive direction.

So the code documents and preserves the full experiment, and the notebook contains prior outputs, but reproducing it locally requires obtaining the external data and replacing the hard-coded paths.
---
title: "Text Mining Solution"
author: "Sabrina Michl"
date: "2023-01-31"
categories: [code, analysis]
bibliography: ref.bib
image: "image_solution.png"
---


# 1. Preliminary Note & Definition

## 1.1 Preliminary Note

For this analysis we use the dataset from @data/0B5VML_2019 out of the zip archive @data/0B5VML/XIUWJ7_2019. The data are licensed according to Attribution 4.0 International (CC-BY-4.0).

The welcome page picture, to this post is from <a href="https://pixabay.com/de/users/talhakhalil007-5671515/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=4271251">talha khalil</a> at <a href="https://pixabay.com/de//?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=4271251">Pixabay</a>

## 1.2 Definition of Hate Speech

The [United Nations](https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech) defines hate speech as: "***any kind of communication** in speech, writing or behaviour, that **attacks** or uses **pejorative** or **discriminatory** language with reference to a person or a group on the basis of **who they are**, in other words, based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."*

The problem about hate speech is, that there is no universal definition that can be used.

So our research question is, if there is any possibility to predict hate speech by using tweets and how well is the prediction?

# 2. Load The Packages


::: {.cell}

```{.r .cell-code}
library(tidyverse)
library(rio)
library(tidymodels)
library(tidytext)
library(textrecipes)
library(lsa)
library(discrim)
library(naivebayes)
library(tictoc)
library(fastrtext)
library(remoji)
library(tokenizers)
```
:::


# 3. Load Dataset And Minor Changes

## 3.1 Train Dataset


::: {.cell}

```{.r .cell-code}
d_train <- read_tsv("C:/Users/sapi-/OneDrive/Studium/5. Semester/Data Science II/Data_Science_Blog/daten/germeval2018.training.txt", col_names = FALSE)
```
:::


### Rename Columns


::: {.cell}

```{.r .cell-code}
names(d_train) <- c("text", "c1", "c2")
```
:::


### Add ID Column


::: {.cell}

```{.r .cell-code}
d_train <- d_train %>%
mutate(id = row_number()) %>%
select(id, everything())
```
:::


## 3.2 Test Dataset


::: {.cell}

```{.r .cell-code}
d_test <- read_tsv("C:/Users/sapi-/OneDrive/Studium/5. Semester/Data Science II/Data_Science_Blog/daten/germeval2018.test.txt", col_names = FALSE)
```
:::


### Rename Columns


::: {.cell}

```{.r .cell-code}
names(d_test) <- c("text", "c1", "c2")
```
:::


### Add ID Column


::: {.cell}

```{.r .cell-code}
d_test <- d_test %>%
mutate(id = row_number()) %>%
select(id, everything())
```
:::


# 4. Insert The Predefined Word Embedding List

The used word embeddings are from @grave2018learning. The data are licensed according to Attribution-ShareAlike 3.0 Unported (CC-BY-SA 3.0).


::: {.cell}

```{.r .cell-code}
out_file_model <- "C:/Users/sapi-/OneDrive - Hochschule für Angewandte Wissenschaften Ansbach/Desktop/AWM/angewandte Wirtschats- und Medienpsychologie/5. Semester/Word_Embedding/de.300.bin"
```
:::

::: {.cell}

```{.r .cell-code}
file.exists(out_file_model)
```

::: {.cell-output .cell-output-stdout}
```
[1] TRUE
```
:::
:::

::: {.cell}

```{.r .cell-code}
fasttext_model <- load_model(out_file_model)
dictionary <- get_dictionary(fasttext_model)
get_word_vectors(fasttext_model, c("menschen")) %>% `[`(1:10)
```

::: {.cell-output .cell-output-stdout}
```
 [1] -0.043737594 -0.033647023 -0.016398411  0.037433818  0.029863771
 [6] -0.008217440  0.002691153 -0.027484305 -0.058012061  0.004103063
```
:::
:::

::: {.cell}

```{.r .cell-code}
print(head(dictionary, 10))
```

::: {.cell-output .cell-output-stdout}
```
 [1] ","    "."    "</s>" "und"  "der"  ":"    "die"  "\""   ")"    "("   
```
:::
:::

::: {.cell}

```{.r .cell-code}
word_embedding_text <- tibble(word = dictionary)
```
:::

::: {.cell}

```{.r .cell-code}
options(mc.cores = parallel::detectCores())
words_vecs <- get_word_vectors(fasttext_model)
```
:::

::: {.cell}

```{.r .cell-code}
word_embedding_text <-
word_embedding_text %>%
bind_cols(words_vecs)
```
:::

::: {.cell}

```{.r .cell-code}
names(word_embedding_text) <- c("word", paste0("v", sprintf("%03d", 1:301)))
```
:::


# 5. Insert The Helperfunctions

We are using the package [pradadata](https://github.com/sebastiansauer/pradadata) by @sebastian_sauer_2018_1996614. The data are licensed according to General Public License 3 (GLP-3).


::: {.cell}

```{.r .cell-code}
data("schimpwoerter", package = "pradadata")
data("sentiws", package = "pradadata")
data("wild_emojis", package = "pradadata")
source("C:/Users/sapi-/OneDrive/Studium/5. Semester/Data Science II/Data_Science_Blog/helper/helper_funs.R")
```
:::


# 6. Define Recipe - rec4 - TF-IDF


::: {.cell}

```{.r .cell-code}
rec4 <-
recipe(c1 ~., data = select(d_train, text, c1, id)) %>%
update_role(id, new_role = "id") %>% 
step_text_normalization(text) %>%
step_mutate(emo_count = map_int(text, ~count_lexicon(.x, sentiws$word))) %>%
step_mutate(schimpf_count = map_int(text, ~count_lexicon(.x, schimpfwoerter$word))) %>%
step_mutate(wild_emojis = map_int(text, ~count_lexicon(.x, wild_emojis$emoji))) %>%
step_mutate(text_copy = text) %>%
step_textfeature(text_copy) %>%
step_tokenize(text) %>%
step_stopwords(text, language = "de", stopword_source = "snowball") %>%
step_stem(text) %>%
step_tfidf(text)
```
:::

::: {.cell}

```{.r .cell-code}
rec4_prep <- rec4 %>%
prep() %>%
recipes::bake(new_data = NULL)
```
:::


# 7. Build Resamples


::: {.cell}

```{.r .cell-code}
folds <- vfold_cv(data = d_train,
v = 3,
repeats = 2,
strata = c1)
```
:::


# 8. Build the Penalty-Grid


::: {.cell}

```{.r .cell-code}
lambda_grid <- grid_regular(penalty(), levels = 25)
```
:::


# 9. Lasso-L1 With TF-IDF

We take only the best model from the [analysis](https://world-of-datascience.netlify.app/posts/text-mining_new/).

### L1-Model


::: {.cell}

```{.r .cell-code}
l1_86_mod <- logistic_reg(penalty = tune(), mixture = 1) %>%
set_engine("glmnet") %>%
set_mode("classification")
l1_86_mod
```

::: {.cell-output .cell-output-stdout}
```
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = tune()
  mixture = 1

Computational engine: glmnet 
```
:::
:::


### Define The Workflow


::: {.cell}

```{.r .cell-code}
l1_86_wf <-workflow() %>%
add_recipe(rec4) %>%
add_model(l1_86_mod)
l1_86_wf
```

::: {.cell-output .cell-output-stdout}
```
== Workflow ====================================================================
Preprocessor: Recipe
Model: logistic_reg()

-- Preprocessor ----------------------------------------------------------------
10 Recipe Steps

* step_text_normalization()
* step_mutate()
* step_mutate()
* step_mutate()
* step_mutate()
* step_textfeature()
* step_tokenize()
* step_stopwords()
* step_stem()
* step_tfidf()

-- Model -----------------------------------------------------------------------
Logistic Regression Model Specification (classification)

Main Arguments:
  penalty = tune()
  mixture = 1

Computational engine: glmnet 
```
:::
:::


### Resampling & Model Quality


::: {.cell}

```{.r .cell-code}
options(mc.cores = parallel::detectCores())
l1_86_wf_fit <- tune_grid(
l1_86_wf,
folds,
grid = lambda_grid,
control = control_resamples(save_pred = TRUE)
)
```

::: {.cell-output .cell-output-stderr}
```
Warning: Paket 'stringi' wurde unter R Version 4.1.2 erstellt
```
:::

::: {.cell-output .cell-output-stderr}
```
Warning: Paket 'textfeatures' wurde unter R Version 4.1.3 erstellt
```
:::

::: {.cell-output .cell-output-stderr}
```
Warning: Paket 'stopwords' wurde unter R Version 4.1.3 erstellt
```
:::

::: {.cell-output .cell-output-stderr}
```
Warning: Paket 'glmnet' wurde unter R Version 4.1.3 erstellt
```
:::

::: {.cell-output .cell-output-stderr}
```
Warning: Paket 'Matrix' wurde unter R Version 4.1.3 erstellt
```
:::
:::


### Model Performance


::: {.cell}

```{.r .cell-code}
l1_86_wf_performance <- collect_metrics(l1_86_wf_fit)
l1_86_wf_performance
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 50 x 7
    penalty .metric  .estimator  mean     n std_err .config              
      <dbl> <chr>    <chr>      <dbl> <int>   <dbl> <chr>                
 1 1   e-10 accuracy binary     0.746     6 0.00130 Preprocessor1_Model01
 2 1   e-10 roc_auc  binary     0.771     6 0.00431 Preprocessor1_Model01
 3 2.61e-10 accuracy binary     0.746     6 0.00130 Preprocessor1_Model02
 4 2.61e-10 roc_auc  binary     0.771     6 0.00431 Preprocessor1_Model02
 5 6.81e-10 accuracy binary     0.746     6 0.00130 Preprocessor1_Model03
 6 6.81e-10 roc_auc  binary     0.771     6 0.00431 Preprocessor1_Model03
 7 1.78e- 9 accuracy binary     0.746     6 0.00130 Preprocessor1_Model04
 8 1.78e- 9 roc_auc  binary     0.771     6 0.00431 Preprocessor1_Model04
 9 4.64e- 9 accuracy binary     0.746     6 0.00130 Preprocessor1_Model05
10 4.64e- 9 roc_auc  binary     0.771     6 0.00431 Preprocessor1_Model05
# ... with 40 more rows
```
:::
:::

::: {.cell}

```{.r .cell-code}
l1_86_wf_fit_preds <- collect_predictions(l1_86_wf_fit)
```
:::

::: {.cell}

```{.r .cell-code}
l1_86_wf_fit_preds %>% 
  group_by(id) %>% 
  roc_curve(truth = c1, .pred_OFFENSE) %>% 
  autoplot()
```

::: {.cell-output-display}
![](index_files/figure-html/unnamed-chunk-26-1.png){width=672}
:::
:::


### Select The Best


::: {.cell}

```{.r .cell-code}
chosen_auc_l1_86_wf_fit <-
l1_86_wf_fit %>%
select_by_one_std_err(metric = "roc_auc", -penalty)
chosen_auc_l1_86_wf_fit
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 1 x 9
  penalty .metric .estimator  mean     n std_err .config            .best .bound
    <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>              <dbl>  <dbl>
1 0.00825 roc_auc binary     0.783     6 0.00381 Preprocessor1_Mod~ 0.783  0.779
```
:::
:::


# 10. Predictions

## Finalize The Workflow


::: {.cell}

```{.r .cell-code}
l1_86_wf_final <- 
  l1_86_wf %>% 
  finalize_workflow(select_best(l1_86_wf_fit, metric = "roc_auc"))
```
:::


## Adaptation Of The Finished Workflow To The Training Dataset


::: {.cell}

```{.r .cell-code}
options(mc.cores = parallel::detectCores())
fit_train <- l1_86_wf_final %>% 
  fit(d_train)
```
:::


## Adapt The Predictions To The Test Dataset


::: {.cell}

```{.r .cell-code}
fit_test <- fit_train %>% 
  predict(d_test)
```
:::


# 11. Check The Predictions

## Add ID Column


::: {.cell}

```{.r .cell-code}
fit_test <- fit_test %>% 
  mutate(id = row_number())
```
:::


## Join Both Datasets


::: {.cell}

```{.r .cell-code}
test <- fit_test %>% 
  full_join(d_test, by = "id")
```
:::

::: {.cell}

```{.r .cell-code}
test <- test %>% 
  select(id, text, c1, c2, .pred_class)
```
:::


## Test The Predictions


::: {.cell}

```{.r .cell-code}
test %>% 
  count(c1, .pred_class)
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 4 x 3
  c1      .pred_class     n
  <chr>   <fct>       <int>
1 OFFENSE OFFENSE       237
2 OFFENSE OTHER         521
3 OTHER   OFFENSE       113
4 OTHER   OTHER        1357
```
:::
:::

::: {.cell}

```{.r .cell-code}
test %>% 
  filter(c1 == "OTHER", .pred_class == "OTHER") %>% 
  nrow/nrow(test)
```

::: {.cell-output .cell-output-stdout}
```
[1] 0.6090664
```
:::
:::

::: {.cell}

```{.r .cell-code}
test %>% 
  filter(c1 == "OFFENSE", .pred_class == "OFFENSE") %>% 
  nrow/nrow(test)
```

::: {.cell-output .cell-output-stdout}
```
[1] 0.1063734
```
:::
:::


> The table shows us how the predictions `.pred_class` match the actual classified data `c1`.
>
> So we see, that 237 OFFENSE (10,6 %) tweets are actually predicted to be OFFENSE and that 1357 (60,9 %) OTHER classified tweets are predicted to be OTHER.
>
> Only 634 tweets were not predicted correctly.

Accuracy


::: {.cell}

```{.r .cell-code}
accuracy(test, truth = factor(c1), estimate = .pred_class)
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 1 x 3
  .metric  .estimator .estimate
  <chr>    <chr>          <dbl>
1 accuracy binary         0.715
```
:::
:::

::: {.cell}

```{.r .cell-code}
sens(test, truth = factor(c1), estimate = .pred_class)
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 1 x 3
  .metric .estimator .estimate
  <chr>   <chr>          <dbl>
1 sens    binary         0.313
```
:::
:::

::: {.cell}

```{.r .cell-code}
spec(test, truth = factor(c1), estimate = .pred_class)
```

::: {.cell-output .cell-output-stdout}
```
# A tibble: 1 x 3
  .metric .estimator .estimate
  <chr>   <chr>          <dbl>
1 spec    binary         0.923
```
:::
:::


# 12. Research Question

Now we have to clear the research question, it there is any possibility to predict hate speech by using tweets and how well is the prediction?

1.  Yes, we can predict hate speech with tweets!
2.  The accuracy (71,5 %) shows us, how well the prediction is!


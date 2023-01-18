library(readr)
library(stringr)
library(dplyr)

# options(tibble.width = 90)
# options(tibble.n = 90)
options("width"=200)
options(setWidthOnResize=TRUE)


# df <- read_delim("toks.csv")
df <- read_delim("pretok.csv")
# df <- read_delim("swedish.csv")
# df <- read_delim("no-bpe-wordpiece-tok.csv")

df <- df %>%
  select(-c(`otok_ttl/unq`, `ptok_ttl/unq`, form_voc_pref, form_voc_suff)) %>%
  mutate_if(is.character, stringr::str_trim)


colnames(df) <- trimws(colnames(df))
colnames(df) <- stringr::str_replace_all(colnames(df), pattern = " ", replacement = "_")

df <- df %>%
  # mutate_at(vars(otok_unique_tokens:fertility), as.numeric)
  mutate_at(vars(lemma_voc_pref:fertility), as.numeric)

df %>%
  group_by(train_data) %>%
  summarise_at(vars(lemma_voc_pref:fertility), .funs = mean) %>%
  arrange(fertility) %>%
  print(n=50)

df %>%
  group_by(pretok) %>%
  summarise_at(vars(lemma_voc_pref:fertility), .funs = mean) %>%
  arrange(fertility) %>%
  print(n=50)

df %>%
  group_by(size) %>%
  summarise_at(vars(lemma_voc_pref:fertility), .funs = mean) %>%
  arrange(fertility) %>%
  print(n=50)

df %>%
  group_by(type) %>%
  summarise_at(vars(lemma_voc_pref:fertility), .funs = mean) %>%
  arrange(fertility) %>%
  print(n=50)

df %>%
  group_by(type, pretok, size, train_data) %>%
  summarise_at(vars(lemma_voc_pref:fertility), .funs = mean) %>%
  arrange(fertility) %>%
  print(n=50)
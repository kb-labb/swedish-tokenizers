library(readr)
library(stringr)
library(dplyr)

# options(tibble.width = 90)
# options(tibble.n = 90)
options("width"=200)
options(setWidthOnResize=TRUE)


df <- read_delim("tatt-combined.tsv")

df <- df %>%
  mutate_if(is.character, stringr::str_trim)


colnames(df) <- trimws(colnames(df))
colnames(df) <- stringr::str_replace_all(colnames(df), pattern = " ", replacement = "_")

df <- df %>%
  mutate_at(vars(ttl_tok:avg_frtlty), as.numeric)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(train_data) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(pretok) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(eval_data) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type, pretok, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type, pretok) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(pretok, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type, train_data, pretok, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type,  train_data, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type,  train_data, pretok) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(type, train_data) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(train_data, pretok, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(train_data, pretok) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)


df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(train_data, size) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

df %>%
  select(-c(avg_doc_len_ws, avg_doc_len_tok)) %>%
  group_by(train_data, eval_data) %>%
  summarise_at(vars(ttl_tok:avg_frtlty), .funs = mean) %>%
  arrange(avg_frtlty) %>%
  print(n=50)

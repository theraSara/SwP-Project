library(lme4)
library(lmerTest)

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
df <- read.csv("data_output/master_modeling_data.csv")

# Factors
df$SUB <- as.factor(df$SUB)
df$ITEM <- as.factor(df$ITEM)

# Scale continuous variables upfront
df$s_word_length <- scale(df$word_length)
df$s_word_freq   <- scale(df$word_frequency)
df$s_word_pos    <- scale(df$word_position)

df$s_cloze       <- scale(df$cloze_surprisal)

# Scale Unidirectional LLM variables
df$s_gpt2_uni    <- scale(df$gpt2_uni_surprisal)
df$s_g270_uni    <- scale(df$gemma270m_uni_surprisal)
df$s_g12b_uni    <- scale(df$gemma12b_uni_surprisal)

# Scale Bidirectional LLM variables
df$s_gpt2_bi     <- scale(df$gpt2_bi_surprisal)
df$s_g270_bi     <- scale(df$gemma270m_bi_surprisal)
df$s_g12b_bi     <- scale(df$gemma12b_bi_surprisal)


# ==========================================
# 2. RUN BASELINE & HUMAN MODELS
# ==========================================
print("Running Baseline and Human models...")
m_base  <- lmer(log_rt ~ s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_cloze <- lmer(log_rt ~ s_cloze + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)


# ==========================================
# 3. A. RUN LLM MODELS (Unidirectional)
# ==========================================
print("Running LLM models (Unidirectional)...")
m_gpt2 <- lmer(log_rt ~ s_gpt2_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_g270 <- lmer(log_rt ~ s_g270_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_g12b <- lmer(log_rt ~ s_g12b_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)

# ==========================================
# 3. B. RUN LLM MODELS (Bidirectional)
# ==========================================
print("Running LLM models (Bidirectional)...")
m_gpt2_bi <- lmer(log_rt ~ s_gpt2_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_g270_bi <- lmer(log_rt ~ s_g270_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_g12b_bi <- lmer(log_rt ~ s_g12b_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)


# ==========================================
# 4. A. RUN COMBINED MODELS (Human + LLM-Unidirectional)
# ==========================================
print("Running Combined models (Human + LLM-Unidirectional)...")
m_both_gpt2 <- lmer(log_rt ~ s_cloze + s_gpt2_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_both_g270 <- lmer(log_rt ~ s_cloze + s_g270_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_both_g12b <- lmer(log_rt ~ s_cloze + s_g12b_uni + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)

# ==========================================
# 4. B. RUN COMBINED MODELS (Human + LLM-Bidirectional)
# ==========================================
print("Running Combined models (Human + LLM-Bidirectional)...")
m_both_gpt2_bi <- lmer(log_rt ~ s_cloze + s_gpt2_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_both_g270_bi <- lmer(log_rt ~ s_cloze + s_g270_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)
m_both_g12b_bi <- lmer(log_rt ~ s_cloze + s_g12b_bi + s_word_length + s_word_freq + s_word_pos + (1|SUB) + (1|ITEM), data=df, REML=FALSE)


# ==========================================
# 5. PRINT MASTER COMPARISONS TO CONSOLE
# ==========================================
print("---------------------------------------------------------")
print("AIC COMPARISON: UNIDIRECTIONAL (Lower is Better)")
print("---------------------------------------------------------")
print(AIC(m_base, m_cloze, m_gpt2, m_g270, m_g12b))

print("---------------------------------------------------------")
print("AIC COMPARISON: BIDIRECTIONAL (Lower is Better)")
print("---------------------------------------------------------")
print(AIC(m_base, m_cloze, m_gpt2_bi, m_g270_bi, m_g12b_bi))


# ==========================================
# 6. SAVE OUTPUTS TO SEPARATE TEXT FILES
# ==========================================
dir.create("regression_output", showWarnings = FALSE)

# ----------------- GPT-2 -----------------
sink("regression_output/gpt2_effects.txt")
cat("\n================ UNIDIRECTIONAL ================\n")
print(summary(m_gpt2))
print("--- Does GPT-2 (UNI) improve over baseline? ---")
print(anova(m_base, m_gpt2))
print("--- Does GPT-2 (UNI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_gpt2))

cat("\n================ BIDIRECTIONAL ================\n")
print(summary(m_gpt2_bi))
print("--- Does GPT-2 (BI) improve over baseline? ---")
print(anova(m_base, m_gpt2_bi))
print("--- Does GPT-2 (BI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_gpt2_bi))
sink()

# ----------------- GEMMA 270M -----------------
sink("regression_output/gemma270m_effects.txt")
cat("\n================ UNIDIRECTIONAL ================\n")
print(summary(m_g270))
print("--- Does Gemma-270M (UNI) improve over baseline? ---")
print(anova(m_base, m_g270))
print("--- Does Gemma-270M (UNI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_g270))

cat("\n================ BIDIRECTIONAL ================\n")
print(summary(m_g270_bi))
print("--- Does Gemma-270M (BI) improve over baseline? ---")
print(anova(m_base, m_g270_bi))
print("--- Does Gemma-270M (BI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_g270_bi))
sink()

# ----------------- GEMMA 12B -----------------
sink("regression_output/gemma12b_effects.txt")
cat("\n================ UNIDIRECTIONAL ================\n")
print(summary(m_g12b))
print("--- Does Gemma-12B (UNI) improve over baseline? ---")
print(anova(m_base, m_g12b))
print("--- Does Gemma-12B (UNI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_g12b))

cat("\n================ BIDIRECTIONAL ================\n")
print(summary(m_g12b_bi))
print("--- Does Gemma-12B (BI) improve over baseline? ---")
print(anova(m_base, m_g12b_bi))
print("--- Does Gemma-12B (BI) explain variance not captured by humans? ---")
print(anova(m_cloze, m_both_g12b_bi))
sink()

print("All done! Results saved in the 'regression_output' folder.")
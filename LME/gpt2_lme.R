library(lme4)
library(lmerTest)

df <- read.csv("data/bk21_modeling_data.csv")

df$SUB <- as.factor(df$SUB)
df$ITEM <- as.factor(df$ITEM)
df$condition <- as.factor(df$condition)

m_base <- lmer(log_rt ~ word_length + log_freq +
               (1|SUB) + (1|ITEM),
               data=df, REML=FALSE)

summary(m_base)


m_cloze <- lmer(log_rt ~ cloze_surprisal +
                word_length + log_freq +
                (1|SUB) + (1|ITEM),
                data=df, REML=FALSE)

summary(m_cloze)


m_gpt2 <- lmer(log_rt ~ gpt2_s_surprisal +
               word_length + log_freq +
               (1|SUB) + (1|ITEM),
               data=df, REML=FALSE)

summary(m_gpt2)

m_both <- lmer(log_rt ~ cloze_surprisal +
               gpt2_s_surprisal +
               word_length + log_freq +
               (1|SUB) + (1|ITEM),
               data=df, REML=FALSE)

summary(m_both)

anova(m_base, m_cloze)
anova(m_base, m_gpt2)
anova(m_cloze, m_both)

AIC(m_base, m_cloze, m_gpt2, m_both)

sink("output/gpt2_effects.txt")
summary(m_cloze)
anova(m_base,m_cloze)
sink()


# ---------------------------
# Helper: fit model + extract Dunnett contrasts
# Returns a data.frame with columns:
#   contrast, estimate, SE, t_stat
# ---------------------------
set.seed(47)

get_contrasts <- function(df, ref_level = 'llm', contr_method = "trt.vs.ctrl", adj = "dunnett", alpha = 0.05) {
  
  fit <- lm(papers_found ~ condition + dataset + n_trials, data = df)
  
  emm <- emmeans::emmeans(fit, ~ condition)
  
  # Dunnett-style contrasts (treatment vs control) on identity scale
  con <- emmeans::contrast(emm, method = contr_method, ref = ref_level, adjust = "dunnett")
  
  # Extract on the coefficient scale (difference in means)
  out <- as.data.frame(summary(con))
  
  # Standardize column names
  out <- out %>%
    transmute(
      contrast = as.character(contrast),
      estimate = as.numeric(estimate),
      SE       = as.numeric(SE),
      t_stat   = as.numeric(t.ratio),
      p_value = as.numeric(p.value)
    )
  
  out
}

B <- 10000
contrast_levels <- get_contrasts(simulation)$contrast

boot_est <- matrix(NA_real_, nrow = B, ncol = length(contrast_levels), dimnames = list(NULL, contrast_levels))
boot_se  <- matrix(NA_real_, nrow = B, ncol = length(contrast_levels), dimnames = list(NULL, contrast_levels))
boot_t   <- matrix(NA_real_, nrow = B, ncol = length(contrast_levels), dimnames = list(NULL, contrast_levels))
boot_p <- matrix(NA_real_, nrow = B, ncol = length(contrast_levels), dimnames = list(NULL, contrast_levels))

for (i in seq_len(B)){
  
  idx <- sample.int(nrow(simulation), 10000, replace = TRUE)
  
  results <- get_contrasts(simulation[idx,])
  
  boot_est[i, ] <- results$estimate
  boot_se[i, ]  <- results$SE
  boot_t[i, ]   <- results$t_stat
  boot_p[i, ] <- results$p_value
  
  if (i %% 200 == 0) message("Completed ", i, " / ", B)
}


#plot overlapping histograms of the p_values of all three contrasts
hist(boot_est[,1], breaks = 20, col = rgb(1,0,0,0.5), xlim = c(-12,2), main = "Bootstrap estimate", xlab = "estimate") 
hist(boot_est[,2], breaks = 20, col = rgb(0,1,0,0.5), add = TRUE) 
hist(boot_est[,3], breaks = 20, col = rgb(0,0,1,0.5), add = TRUE) 
legend("topright", legend = contrast_levels, fill = c(rgb(1,0,0,0.5), rgb(0,1,0,0.5), rgb(0,0,1,0.5)))

var(boot_est[,1])
var(boot_est[,2])
var(boot_est[,3])
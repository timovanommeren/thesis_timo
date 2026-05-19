
# =============================================================================
# bootstrap_groupmeans.R
#
# Sourcing this file adds five functions to the global workspace:
#
#   bootstrap_group_means()          – runs the bootstrap, returns a list
#   plot_group_means_boot()          – base R: coloured boxplot
#   plot_group_means_boot_ggplot()   – ggplot2: boxplot + CI error bars
#   plot_group_means_boot_ggdist()   – ggdist: half-eye density + CI
#   plot_group_means_boot_beeswarm() – ggbeeswarm: jittered draws + CI
#
# No code is executed on source.
# =============================================================================


# ── 1. Bootstrap ──────────────────────────────────────────────────────────────
#
# Fits lm(papers_found ~ condition - 1 + dataset + n_trials) on `df` and on
# B bootstrap resamples. The no-intercept formula means each condition
# coefficient is directly the covariate-adjusted group mean.
#
# Parameters
# ----------
# df        : data.frame – simulation data (papers_found, condition, dataset,
#                          n_trials)
# B         : integer    – number of bootstrap replicates  (default 10 000)
# seed      : integer    – RNG seed for reproducibility    (default 47)
# ci_level  : numeric    – confidence level for percentile CI (default 0.95)
# verbose   : logical    – print progress every 500 iterations (default TRUE)
#
# Returns
# -------
# A list with:
#   $boot_summary  data.frame – condition | obs_mean | boot_se | ci_lo | ci_hi
#   $boot_means    matrix     – B × k matrix of per-replicate group means
# -----------------------------------------------------------------------------
library(emmeans)

bootstrap_group_means <- function(df,
                                  B        = 10000L,
                                  seed     = 47L,
                                  ci_level = 0.95,
                                  verbose  = TRUE) {
  set.seed(seed)
  
  # inner helper: fit model, return named vector of condition emmeans
  .get_group_means <- function(d) {
    fit  <- lm(papers_found ~ condition - 1 + dataset + n_trials, data = d)
    emm  <- emmeans(fit, ~ condition)
    smm  <- summary(emm)
    cond <- setNames(smm$emmean, as.character(smm$condition))
    cond
  }
  
  obs_means    <- .get_group_means(df)
  group_levels <- names(obs_means)
  k            <- length(group_levels)
  n            <- nrow(df)
  
  boot_means <- matrix(
    NA_real_,
    nrow     = B,
    ncol     = k,
    dimnames = list(NULL, group_levels)
  )
  
  for (i in seq_len(B)) {
    idx            <- sample.int(n, n, replace = TRUE)
    boot_means[i,] <- .get_group_means(df[idx, ])
    if (verbose && i %% 500 == 0)
      message("Completed ", i, " / ", B)
  }
  
  alpha   <- 1 - ci_level
  boot_se <- apply(boot_means, 2, sd,                               na.rm = TRUE)
  boot_lo <- apply(boot_means, 2, quantile, probs = alpha / 2,     na.rm = TRUE)
  boot_hi <- apply(boot_means, 2, quantile, probs = 1 - alpha / 2, na.rm = TRUE)
  
  boot_summary <- data.frame(
    condition = group_levels,
    obs_mean  = obs_means[group_levels],
    boot_se   = boot_se[group_levels],
    ci_lo     = boot_lo[group_levels],
    ci_hi     = boot_hi[group_levels],
    row.names = NULL
  )

  # ── Pairwise comparisons ───────────────────────────────────────────────────
  # For each pair (g1, g2):
  #   obs_diff  – observed difference in model-estimated means (g1 − g2)
  #   ci_lo/hi  – percentile CI of the bootstrap difference distribution
  #   p_value   – two-tailed bootstrap p-value (reflection method):
  #               the bootstrap distribution of (g1 − g2) is shifted to be
  #               centred at 0 (H₀), then the p-value is the fraction of
  #               shifted draws at least as extreme as the observed difference
  pairs <- utils::combn(group_levels, 2, simplify = FALSE)

  pairwise_list <- lapply(pairs, function(pr) {
    g1     <- pr[1]; g2 <- pr[2]
    d_obs  <- obs_means[g1] - obs_means[g2]
    d_boot <- boot_means[, g1] - boot_means[, g2]

    d_centered <- d_boot - mean(d_boot, na.rm = TRUE)
    p_val      <- mean(abs(d_centered) >= abs(d_obs), na.rm = TRUE)

    ci_d_lo <- quantile(d_boot, probs = alpha / 2,     na.rm = TRUE)
    ci_d_hi <- quantile(d_boot, probs = 1 - alpha / 2, na.rm = TRUE)

    data.frame(
      condition1 = g1,
      condition2 = g2,
      obs_diff   = unname(d_obs),
      ci_lo      = unname(ci_d_lo),
      ci_hi      = unname(ci_d_hi),
      p_value    = p_val,
      stringsAsFactors = FALSE
    )
  })

  pairwise_comparisons <- do.call(rbind, pairwise_list)
  row.names(pairwise_comparisons) <- NULL

  list(
    boot_summary         = boot_summary,
    boot_means           = boot_means,
    pairwise_comparisons = pairwise_comparisons
  )
}


# ── 2. Null-distribution diagnostic plots ────────────────────────────────────
#
# For each pairwise comparison, plots the shifted bootstrap distribution
# (d_centered, i.e. the null distribution) as a histogram, with the observed
# difference (d_obs) and its mirror (−d_obs) marked as vertical lines.
# The shaded area shows the fraction of draws that contributed to the p-value.
#
# Parameters
# ----------
# boot_result : list    – returned by bootstrap_group_means()
# labels      : named character vector – condition label per condition name
#               (defaults to condition names if NULL)
# ncol        : integer – columns in the facet grid (default 3)
# -----------------------------------------------------------------------------
plot_boot_null_distributions <- function(
    boot_result,
    labels = NULL,
    ncol   = 3L) {

  pw         <- boot_result$pairwise_comparisons
  boot_means <- boot_result$boot_means

  # build long data.frame: one row per bootstrap draw per comparison
  plot_list <- lapply(seq_len(nrow(pw)), function(i) {
    g1     <- pw$condition1[i]
    g2     <- pw$condition2[i]
    d_boot <- boot_means[, g1] - boot_means[, g2]

    d_centered <- d_boot - mean(d_boot, na.rm = TRUE)
    d_obs      <- pw$obs_diff[i]

    lbl1 <- if (!is.null(labels) && g1 %in% names(labels)) labels[g1] else g1
    lbl2 <- if (!is.null(labels) && g2 %in% names(labels)) labels[g2] else g2

    data.frame(
      comparison  = paste0(lbl1, " \u2212 ", lbl2),   # "A − B"
      d_centered  = d_centered,
      d_obs       = d_obs,
      p_value     = pw$p_value[i],
      stringsAsFactors = FALSE
    )
  })

  long_df <- do.call(rbind, plot_list)

  # one label row per facet (for geom_vline / geom_text)
  facet_df <- unique(long_df[, c("comparison", "d_obs", "p_value")])
  facet_df$p_label <- ifelse(
    facet_df$p_value < .001,
    "p < .001",
    sprintf("p = %.3f", facet_df$p_value)
  )

  p <- ggplot2::ggplot(long_df, ggplot2::aes(x = d_centered)) +
    ggplot2::geom_histogram(
      ggplot2::aes(fill = abs(d_centered) >= abs(d_obs)),
      bins   = 60,
      colour = NA
    ) +
    ggplot2::geom_vline(
      data      = facet_df,
      ggplot2::aes(xintercept =  d_obs),
      colour    = "#D55E00", linewidth = 0.9, linetype = "solid"
    ) +
    ggplot2::geom_vline(
      data      = facet_df,
      ggplot2::aes(xintercept = -d_obs),
      colour    = "#D55E00", linewidth = 0.9, linetype = "dashed"
    ) +
    ggplot2::geom_text(
      data  = facet_df,
      ggplot2::aes(
        x     = -Inf,
        y     = Inf,
        label = p_label
      ),
      hjust  = -0.1, vjust = 1.5,
      size   = 3.2,
      colour = "grey20",
      inherit.aes = FALSE
    ) +
    ggplot2::scale_fill_manual(
      values = c("FALSE" = "grey70", "TRUE" = "#0072B2"),
      labels = c("FALSE" = "not as extreme as d_obs",
                 "TRUE"  = "|d_centered| \u2265 |d_obs|  (contributes to p)"),
      name   = NULL
    ) +
    ggplot2::facet_wrap(~ comparison, ncol = ncol, scales = "free") +
    ggplot2::labs(
      x     = "d_centered  (null distribution of the difference)",
      y     = "Count",
      title = "Bootstrap null distributions for pairwise comparisons",
      subtitle = paste0(
        "Histogram: d_boot shifted to \u03bc = 0  |  ",
        "Solid line: +d_obs  |  Dashed line: \u2212d_obs  |  ",
        "Blue bars contribute to p-value"
      )
    ) +
    ggplot2::theme_bw(base_size = 11) +
    ggplot2::theme(
      legend.position  = "bottom",
      strip.text       = ggplot2::element_text(size = 9),
      plot.subtitle    = ggplot2::element_text(size = 8, colour = "grey40")
    )

  print(p)
  invisible(p)
}


# ── Internal helper ────────────────────────────────────────────────────────────
# Shared by all four plot functions.
# Resolves plot order, colours, labels; reshapes boot_means to long format.
.prep_boot_plot_data <- function(boot_result, colours, labels) {

  boot_summary <- boot_result$boot_summary
  boot_means   <- boot_result$boot_means
  group_levels <- boot_summary$condition

  # order follows the colours vector; only keep conditions present in both
  plot_order <- intersect(names(colours), group_levels)
  plot_cols  <- colours[plot_order]
  plot_lbls  <- labels[plot_order]

  # long format of bootstrap draws, restricted to plot_order
  long_df <- tidyr::pivot_longer(
    as.data.frame(boot_means[, plot_order, drop = FALSE]),
    cols      = tidyr::everything(),
    names_to  = "condition",
    values_to = "boot_mean"
  )
  long_df$condition <- factor(long_df$condition, levels = plot_order)

  # boot_summary in matching order, condition as factor
  boot_summary_ord <- boot_summary[match(plot_order, boot_summary$condition), ]
  boot_summary_ord$condition <- factor(boot_summary_ord$condition,
                                       levels = plot_order)

  list(
    boot_summary = boot_summary_ord,
    long_df      = long_df,
    plot_order   = plot_order,
    plot_cols    = plot_cols,
    plot_lbls    = plot_lbls,
    B            = nrow(boot_means)
  )
}


# ── 2. Base R boxplot ─────────────────────────────────────────────────────────
#
# Parameters
# ----------
# boot_result : list    – returned by bootstrap_group_means()
# colours     : named character vector – fill colour per condition
# labels      : named character vector – x-axis label per condition
# ylab        : character – y-axis label
# main        : character – plot title (NULL = auto)
# outline     : logical   – show outlier points (default FALSE)
# save_path   : character – directory to save PNG to (NULL = don't save)
# filename    : character – PNG filename (default "group_means_bootstrap_base.png")
# width/height: numeric   – plot dimensions in inches
# res         : integer   – resolution in DPI
# -----------------------------------------------------------------------------
plot_group_means_boot_ggdist <- function(
    boot_result,
    colours = c(
      llm               = "#0072B2",
      criteria          = "#D55E00",
      random            = "#009E73",
      no_initialisation = "#E69F00"
    ),
    labels = c(
      llm               = "LLM",
      criteria          = "Criteria",
      random            = "Random",
      no_initialisation = "No initialisation"
    ),
    ylab      = "Number of relevant papers found",
    main      = NULL,
    outline   = FALSE,
    save_path = NULL,
    filename  = "group_means_bootstrap_base.png",
    width     = 8,
    height    = 6,
    res       = 300) {

  d <- .prep_boot_plot_data(boot_result, colours, labels)

  if (is.null(main))
    main <- sprintf(
      "Model-estimated group means with bootstrapped uncertainty\n(B = %s)",
      format(d$B, big.mark = ",")
    )

  boot_list        <- lapply(d$plot_order, function(g) boot_result$boot_means[, g])
  names(boot_list) <- d$plot_order

  if (!is.null(save_path))
    png(file.path(save_path, filename),
        width = width, height = height, units = "in", res = res)

  boxplot(
    boot_list,
    col     = d$plot_cols,
    border  = "grey30",
    names   = d$plot_lbls,
    ylab    = ylab,
    main    = main,
    las     = 1,
    outline = outline
  )

  obs <- setNames(d$boot_summary$obs_mean, as.character(d$boot_summary$condition))
  points(
    x   = seq_along(d$plot_order),
    y   = obs[d$plot_order],
    pch = 18, cex = 1.6, col = "black"
  )

  legend(
    "bottomleft",
    legend = c("Model estimate", "Bootstrap IQR / 95% CI"),
    pch    = c(18, 15),
    col    = c("black", "grey60"),
    pt.cex = c(1.6, 1.6),
    bty    = "n"
  )

  if (!is.null(save_path)) dev.off()

  invisible(boot_result)
}


# ── 3. ggplot2 boxplot ────────────────────────────────────────────────────────
#
# Standard ggplot2 boxplot of the bootstrap distribution, with the
# model-estimated mean overlaid as a diamond and 95% CI error bars.
#
# Parameters: same as plot_group_means_boot() except no `outline` argument.
# Additional:
#   filename defaults to "group_means_bootstrap_ggplot.png"
# -----------------------------------------------------------------------------
plot_group_means_boot_ggplot <- function(
    boot_result,
    colours = c(
      llm               = "#0072B2",
      criteria          = "#D55E00",
      random            = "#009E73",
      no_initialisation = "#E69F00"
    ),
    labels = c(
      llm               = "LLM",
      criteria          = "Criteria",
      random            = "Random",
      no_initialisation = "No initialisation"
    ),
    ylab      = "Number of relevant papers found",
    main      = NULL,
    save_path = NULL,
    filename  = "group_means_bootstrap_ggplot.png",
    width     = 8,
    height    = 6,
    res       = 300) {
  d <- .prep_boot_plot_data(boot_result, colours, labels)
  if (is.null(main))
    main <- sprintf(
      "Model-estimated group means with bootstrapped uncertainty\n(B = %s)",
      format(d$B, big.mark = ",")
    )
  
  # ── significance annotations ──────────────────────────────────────────────
  y_tops <- tapply(d$long_df$boot_mean, d$long_df$condition, max, na.rm = TRUE)
  nudge  <- diff(range(d$long_df$boot_mean, na.rm = TRUE)) * 0.03
  
  annot_df <- data.frame(
    condition = c("criteria",     "no_initialisation", "llm"),
    label     = c("*  +",         "*",                  "+"),
    y         = c(y_tops["criteria"] + nudge,
                  y_tops["no_initialisation"] + nudge,
                  y_tops["llm"] + nudge)
  )
  # ─────────────────────────────────────────────────────────────────────────
  
  dummy_lgd <- d$boot_summary[1L, , drop = FALSE]
  dummy_lgd[, c("obs_mean", "ci_lo", "ci_hi")] <- NA_real_
  
  p <- ggplot2::ggplot(
    d$long_df,
    ggplot2::aes(x = condition, y = boot_mean, fill = condition)
  ) +
    ggplot2::geom_boxplot(
      outlier.shape = NA,
      alpha         = 0.7,
      colour        = "grey30"
    ) +
    ggplot2::geom_errorbar(
      data        = d$boot_summary,
      ggplot2::aes(x = condition, ymin = ci_lo, ymax = ci_hi, y = obs_mean,
                   colour = "ci95"),
      width       = 0.15,
      linewidth   = 0.8,
      inherit.aes = FALSE,
      key_glyph   = "vpath"
    ) +
    ggplot2::geom_point(
      data        = d$boot_summary,
      ggplot2::aes(x = condition, y = obs_mean, colour = "model_mean"),
      shape       = 18,
      size        = 3.5,
      inherit.aes = FALSE,
      key_glyph   = "point"
    ) +
    ggplot2::geom_point(
      data        = dummy_lgd,
      ggplot2::aes(x = condition, y = obs_mean, colour = "median"),
      na.rm       = TRUE,
      inherit.aes = FALSE,
      key_glyph   = "path"
    ) +
    ggplot2::geom_point(
      data        = dummy_lgd,
      ggplot2::aes(x = condition, y = obs_mean, colour = "iqr"),
      na.rm       = TRUE,
      inherit.aes = FALSE,
      key_glyph   = "rect"
    ) +
    # ── annotation layer ──────────────────────────────────────────────────
    ggplot2::geom_text(
      data        = annot_df,
      ggplot2::aes(x = condition, y = y, label = label),
      size        = 5,
      inherit.aes = FALSE
    ) +
    # ──────────────────────────────────────────────────────────────────────
    ggplot2::scale_fill_manual(values = d$plot_cols, guide = "none") +
    ggplot2::scale_colour_manual(
      name   = NULL,
      values = c(
        model_mean = "black",
        median     = "grey30",
        iqr        = "grey50",
        ci95       = "black"
      ),
      breaks = c("model_mean", "median", "iqr", "ci95"),
      labels = c(
        model_mean = "Model predicted mean",
        median     = "Bootstrap median",
        iqr        = "Bootstrap IQR (50%)",
        ci95       = "95% bootstrapped CI"
      ),
      guide = ggplot2::guide_legend(
        override.aes = list(
          shape     = c(18,      NA,       NA,       NA     ),
          linetype  = c(0,       1,        0,        1      ),
          fill      = c(NA,      NA,       "grey60", NA     ),
          linewidth = c(NA,      1.3,      NA,       0.8    ),
          size      = c(3.5,     NA,       NA,       NA     ),
          colour    = c("black", "grey30", "grey30", "black")
        )
      )
    ) +
    ggplot2::scale_x_discrete(labels = d$plot_lbls) +
    ggplot2::labs(x = NULL, y = ylab, title = main) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "right",
      plot.title      = ggplot2::element_text(size = 11, hjust = 0.5)
    )
  
  print(p)
  if (!is.null(save_path))
    ggplot2::ggsave(file.path(save_path, filename),
                    plot = p, width = width, height = height, dpi = res)
  invisible(p)
}


# ── 4. ggdist half-eye ────────────────────────────────────────────────────────
#
# Uses ggdist::stat_halfeye() to draw a density slab alongside a point
# interval, which together convey the full bootstrap distribution.
# The model-estimated mean is overlaid as a filled diamond.
#
# Parameters: same as plot_group_means_boot_ggplot() plus:
#   ci_widths : numeric vector – interval widths drawn by the half-eye
#               (default c(0.66, 0.95) shows both 66% and 95% intervals)
#   filename  : defaults to "group_means_bootstrap_ggdist.png"
# -----------------------------------------------------------------------------
plot_group_means_boot_ggdist <- function(
    boot_result,
    colours = c(
      llm               = "#0072B2",
      criteria          = "#D55E00",
      random            = "#009E73",
      no_initialisation = "#E69F00"
    ),
    labels = c(
      llm               = "LLM",
      criteria          = "Criteria",
      random            = "Random",
      no_initialisation = "No initialisation"
    ),
    ci_widths = c(0.66, 0.95),
    ylab      = "Number of relevant papers found",
    main      = NULL,
    save_path = NULL,
    filename  = "group_means_bootstrap_ggdist.png",
    width     = 8,
    height    = 6,
    res       = 300) {

  d <- .prep_boot_plot_data(boot_result, colours, labels)

  if (is.null(main))
    main <- sprintf(
      "Model-estimated group means with bootstrapped uncertainty\n(B = %s)",
      format(d$B, big.mark = ",")
    )

  p <- ggplot2::ggplot(
    d$long_df,
    ggplot2::aes(x = condition, y = boot_mean, fill = condition)
  ) +
    ggdist::stat_halfeye(
      .width       = ci_widths,
      point_colour = NA,       # suppress halfeye's own median point
      slab_alpha   = 0.75,
      normalize    = "groups"  # density scaled within each group separately
    ) +
    ggplot2::geom_point(
      data        = d$boot_summary,
      ggplot2::aes(x = condition, y = obs_mean),
      shape       = 18,
      size        = 3.5,
      colour      = "black",
      inherit.aes = FALSE
    ) +
    ggplot2::scale_fill_manual(values = d$plot_cols) +
    ggplot2::scale_x_discrete(labels = d$plot_lbls) +
    ggplot2::labs(x = NULL, y = ylab, title = main) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "none",
      plot.title      = ggplot2::element_text(size = 11, hjust = 0.5)
    )

  print(p)

  if (!is.null(save_path))
    ggplot2::ggsave(file.path(save_path, filename),
                    plot = p, width = width, height = height, dpi = res)

  invisible(p)
}


# ── 5. ggbeeswarm jitter ──────────────────────────────────────────────────────
#
# Shows individual (subsampled) bootstrap draws as a quasirandom jitter cloud,
# with the model-estimated mean and 95% CI overlaid.
# Subsampling is necessary because B = 10 000 points per group would be
# too dense to render meaningfully.
#
# Parameters: same as plot_group_means_boot_ggplot() plus:
#   n_display : integer – bootstrap draws to show per group (default 300)
#   filename  : defaults to "group_means_bootstrap_beeswarm.png"
# -----------------------------------------------------------------------------
plot_group_means_boot_beeswarm <- function(
    boot_result,
    colours = c(
      llm               = "#0072B2",
      criteria          = "#D55E00",
      random            = "#009E73",
      no_initialisation = "#E69F00"
    ),
    labels = c(
      llm               = "LLM",
      criteria          = "Criteria",
      random            = "Random",
      no_initialisation = "No initialisation"
    ),
    n_display = 300,
    ylab      = "Number of relevant papers found",
    main      = NULL,
    save_path = NULL,
    filename  = "group_means_bootstrap_beeswarm.png",
    width     = 8,
    height    = 6,
    res       = 300) {

  d <- .prep_boot_plot_data(boot_result, colours, labels)

  if (is.null(main))
    main <- sprintf(
      "Model-estimated group means with bootstrapped uncertainty\n(B = %s, %s draws shown per group)",
      format(d$B, big.mark = ","),
      format(min(n_display, d$B), big.mark = ",")
    )

  # subsample per condition for readability
  sub_df <- d$long_df |>
    dplyr::group_by(condition) |>
    dplyr::slice_sample(n = min(n_display, d$B)) |>
    dplyr::ungroup()

  p <- ggplot2::ggplot(
    sub_df,
    ggplot2::aes(x = condition, y = boot_mean, colour = condition)
  ) +
    ggbeeswarm::geom_quasirandom(alpha = 0.35, size = 0.9) +
    ggplot2::geom_errorbar(
      data        = d$boot_summary,
      ggplot2::aes(x = condition, ymin = ci_lo, ymax = ci_hi, y = obs_mean),
      width       = 0.15,
      linewidth   = 0.9,
      colour      = "black",
      inherit.aes = FALSE
    ) +
    ggplot2::geom_point(
      data        = d$boot_summary,
      ggplot2::aes(x = condition, y = obs_mean),
      shape       = 18,
      size        = 4,
      colour      = "black",
      inherit.aes = FALSE
    ) +
    ggplot2::scale_colour_manual(values = d$plot_cols) +
    ggplot2::scale_x_discrete(labels = d$plot_lbls) +
    ggplot2::labs(x = NULL, y = ylab, title = main) +
    ggplot2::theme_bw(base_size = 12) +
    ggplot2::theme(
      legend.position = "none",
      plot.title      = ggplot2::element_text(size = 10, hjust = 0.5)
    )

  print(p)

  if (!is.null(save_path))
    ggplot2::ggsave(file.path(save_path, filename),
                    plot = p, width = width, height = height, dpi = res)

  invisible(p)
}

library(ggplot2)

descriptive_tables <- function(data) {

  #Summary of number of papers found by condition
  desc_by_cond <- data %>%
    group_by(condition) %>%
    summarise(
      papers_found_mean = mean(papers_found),
      papers_found_sd   = sd(papers_found),
      n       = n(),
      .groups = "drop"
    )

  #Summary of

  # Return results as a list
  list(by_condition = desc_by_cond)
}

descriptive_barchart <- function(data, metadata, variable, y_axis = 100, colours = colours) {
  var_quo  <- enquo(variable)          # capture column
  var_name <- rlang::as_name(var_quo)         # e.g. "papers_found"
  
  # Step 1: mean per dataset x condition
  means <- data %>%
    group_by(dataset, condition) %>%
    summarise(
      mean_value = mean(!!var_quo),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(names_from = condition, values_from = mean_value) %>%
    mutate(contrast = .data[["llm"]] - .data[["no_initialisation"]]) %>%
    dplyr::select(dataset, contrast)
  
  # Step 2: join contrast back to full data
  data_with_contrast <- data %>% left_join(means, by = "dataset")
  
  # Step 3: compute mean and SE per dataset x condition
  plot_data <- data_with_contrast %>%
    group_by(dataset, condition, contrast) %>%
    summarise(
      n = n(),
      variable_mean = mean(!!var_quo),
      variable_se   = sd(!!var_quo) / sqrt(n),
      .groups = "drop"
    ) %>%
    tidyr::replace_na(list(variable_se = 0))
  
  plot_data <- plot_data %>%
    left_join(metadata %>% dplyr::select(dataset, percent_rel), by = "dataset") %>%
    mutate(
      dataset = forcats::fct_reorder(
        paste0(dataset, " (", percent_rel, "%)"),
        percent_rel,
        .desc = TRUE
      ),
      condition = factor(
        condition,
        levels = c("llm", "criteria", "random", "no_initialisation")
      )
    )
  
  
  plot <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = dataset, y = variable_mean, fill = factor(condition))
  ) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8), width = 0.7) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = variable_mean - variable_se,
          ymax = variable_mean + variable_se),
      position = ggplot2::position_dodge(width = 0.8),
      width = 0.2
    ) +
    ggplot2::geom_hline(yintercept = y_axis, linetype = "dashed", color = "black", linewidth = 0.4) +
    ggplot2::labs(
      title = "",
      x = "Datasets <span style='color:#888888;'>(ordered by percent_rel of relevant records)</span>",
      y = "",
      fill = "Examples given before screening:"
    ) +
    ggplot2::scale_fill_manual(
      values = colours,
      breaks = c("llm", "criteria", "random", "no_initialisation"),
      labels = c("LLM", "Eligibility criteria", "True examples", "Cold Start")
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title    = element_text(hjust = 0.5, face = "bold"),
      axis.title.x  = ggtext::element_markdown(),
      axis.title.y  = ggtext::element_markdown(),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key       = element_rect(fill = "white", color = NA),
      plot.margin = margin(10, 20, 10, 10),
      axis.text.x = element_text(angle = 55, hjust = 1),
      legend.position = "top"
    )
  
  ggplot2::ggsave(
    filename = here::here(paste0("Report/results/", var_name, "_barchart.png")),
    plot = plot,
    width = 10,
    height = 6,
    dpi = 300
  )
  
  plot
}


descriptive_histogram <- function(data, colours) {
  

  # Ensure consistent ordering (and consistent legend) across plots
  data <- data %>%
    dplyr::mutate(
      condition = factor(condition,
                         levels = c("no_initialisation", "llm", "criteria", "random"))
    )
  
  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = papers_found / n_trials, fill = condition)
  ) +
    geom_histogram(position = "dodge", bins = 30, alpha = 0.7, boundary = 0) +
    coord_cartesian(xlim = c(0, 1)) +
    scale_x_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0))) +
    ggplot2::scale_fill_manual(
      values = colours,
      breaks = names(colours),
      labels = c("LLM", "Eligibility criteria", "True examples", "Cold Start"),
      drop = FALSE
    ) +
    ggplot2::labs(
      title = "Distribution of number of relevant records found by condition",
      x = "Relevant records found (X) divided by number of trials (n <= 100)",
      y = "Count",
      fill = "Condition"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.background  = ggplot2::element_rect(fill = "white", color = NA),
      legend.background = ggplot2::element_rect(fill = "white", color = NA),
      legend.key = ggplot2::element_rect(fill = "white", color = NA),
      plot.margin = ggplot2::margin(10, 20, 10, 10)
    )
  
  ggplot2::ggsave(
    filename = here::here("Report/results/papers_found_histogram.png"),
    plot = plot + ggplot2::labs(title = NULL),   # remove title in saved file
    width = 10, height = 6, dpi = 300
  )
  
  # 1) Build a list of plots (one per dataset)
  dataset_names <- unique(data$dataset)
  
  plots <- lapply(dataset_names, function(dataset_name) {
    ggplot2::ggplot(
      dplyr::filter(data, dataset == dataset_name),
      ggplot2::aes(x = papers_found / n_trials, fill = condition)
    ) +
      ggplot2::geom_histogram(position = "dodge", bins = 30, alpha = 0.7) +
      ggplot2::scale_fill_manual(
        values = colours,
        breaks = names(colours),
        labels = c("LLM", "Eligibility criteria", "True examples", "Cold Start"),
        drop = FALSE
      ) +
      ggplot2::labs(
        title = paste("N relevant records found by condition -", dataset_name),
        x = "Relevant records found (X) divided by number of trials (n <= 100)",
        y = "Count",
        fill = "Condition"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
        panel.background = ggplot2::element_rect(fill = "white", color = NA),
        plot.background  = ggplot2::element_rect(fill = "white", color = NA),
        legend.background = ggplot2::element_rect(fill = "white", color = NA),
        legend.key = ggplot2::element_rect(fill = "white", color = NA),
        plot.margin = ggplot2::margin(10, 20, 10, 10)
      )
  })
  
  # 2) Write to a multi-page PDF with 4 plots per page
  grDevices::pdf(here::here("Report/results/papers_found_histograms.pdf"),
                 width = 11, height = 8.5, onefile = TRUE)
  
  for (i in seq(1, length(plots), by = 4)) {
    page_plots <- plots[i:min(i + 3, length(plots))]
    print(patchwork::wrap_plots(page_plots, ncol = 2, nrow = 2))
  }
  
  grDevices::dev.off()
  
  plot
}



check_homogeneity <- function(data,
                              colours, 
                              condition_col = "condition",
                              papers_found_col = "papers_found",
                              n_trials_col = "n_trials",
                              jitter_width = 0.1,
                              jitter_alpha = 0.3) {
  stopifnot(is.data.frame(data))
  cols_needed <- c(condition_col, papers_found_col, n_trials_col)
  missing_cols <- setdiff(cols_needed, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing columns in `data`: ", paste(missing_cols, collapse = ", "))
  }
  
  # create ratio (and keep in a temp column)
  df <- data
  df[[".ratio"]] <- df[[papers_found_col]] / df[[n_trials_col]]
  
  # boxplot + jitter
  p <- ggplot2::ggplot(
    df,
    ggplot2::aes(
      x = .data[[condition_col]],
      y = .data[[".ratio"]],
      fill = .data[[condition_col]]   # <- key line
    )
  ) +
    ggplot2::geom_boxplot() +
    ggplot2::scale_fill_manual(
      values = colours,
      breaks = names(colours),
      labels = c("LLM", "Eligibility criteria", "True examples", "Cold Start"),
      drop = FALSE
    ) +
    ggplot2::geom_jitter(width = jitter_width, alpha = jitter_alpha)
  
  
  # Levene's test
  lev <- car::leveneTest(stats::as.formula(paste0(".ratio ~ ", condition_col)), data = df)
  
  # variances by condition
  vars <- dplyr::group_by(df, .data[[condition_col]]) %>%
    dplyr::summarise(
      variance = stats::var(.data[[".ratio"]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::rename(!!condition_col := 1)
  
  list(
    plot = p,
    levene = lev,
    variances = vars
  )
}



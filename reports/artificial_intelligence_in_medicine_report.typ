= Artificial Intelligence in Medicine
== Final Report
#datetime(day: 28, month: 07, year: 2025,).display()
=== Artificial Intelligence

Articles with the "Artificial Intelligence" MeSH Term total to approximately 150,000 articles.
#image("./figures/ARTIFICIAL_INTELLIGENCE/rows_per_year.png")

Inflection is at citations = 23
#image("./figures/ARTIFICIAL_INTELLIGENCE/elbow_curve.png")

==== Community detection

239 communities were detected in the network of citations. One will notice that several are nodes of degree 0. This is because this particular community detection graph is run over all nodes with over 23 citations. That means that all nodes of degree zero are cited > 23 times, but are not cited by other articles with over 23 citations.

#image("./figures/ARTIFICIAL_INTELLIGENCE/community_detection.png")

==== Community Composition over Time

This stacked bar chart shows the frequency of articles in a given community and in a given year. One can see that the community `assisted, laparoscopic, robot, robotic, surgery` first came of relatively major relevance in 1999 and lost almost all representation by 2015. However, I think it is more interesting to note, that early articles in artificial intelligence tend to come from the construction of early neural networks in the 1990s. Then, there is a period of time where artificial intelligence is certainly used, but is certainly not is prevalent as it is today. In 2016, a dramatic shift, specifically towards "deep learning" occurred, where "fundamental" is built upon in applications such as cancer screening, medical imagery, and drug discovery.

#image("./figures/ARTIFICIAL_INTELLIGENCE/community_evolution_by_year.png")

==== Geographic Distribution

After last meeting where we discussed the odd density over Madagascar, I found that it is in fact Port au Prince and Reunion that is causing that shift over the ocean.

#image("./figures/ARTIFICIAL_INTELLIGENCE/map_scatter_output.png")

This, then is a Kernel Density Estimate (KDE) of where citations are located, and colored by year.

#image("./figures/ARTIFICIAL_INTELLIGENCE/global_kde_heatmap_by_year.png")

The way to interpret this is that darker colors represent earlier representation in the Artificial Intelligence MeSH Term. Once can see that Western Europe, Northeast United States, and East Asia all of very high and early density. However, South Asia and the Middle East are highly dense, but come later in time.

==== Funding Agency

The various directorates and agencies of the NIH far outstrip any other grant-making group in terms of citations attributed to grants. This indicates that the NIH funds high-impact research.

#image("./figures/ARTIFICIAL_INTELLIGENCE/top_5_agencies_by_year.png")

==== Interdisciplinarity Analysis

As would be generally expected, articles with more citations tend to be cited by articles that have a greater Jaccard distance in MeSH Terms. Specifically, there is a moderately strong positive correlation between the number of citations and the Jaccard between the MeSH Terms of two articles with a Pearson correlation of ~ 0.4

#image("./figures/ARTIFICIAL_INTELLIGENCE/interdisciplinary_nodes_plot.png")

#image("./figures/ARTIFICIAL_INTELLIGENCE/jaccard_vs_citations_scatter.png")

=== Gene Expression

Articles with the "Gene Expression" MeSH Term total to approximately 160,000 articles.
#image("./figures/GENE_EXPRESSION/rows_per_year.png")

#image("./figures/GENE_EXPRESSION/elbow_curve_with_inflection_point.png")

Inflection point is at citations = 25

==== Community Detection

#image("./figures/GENE_EXPRESSION/community_detection.png")

==== Community Composition over Time

#image("./figures/GENE_EXPRESSION/community_evolution_by_year.png")

The community representation follows a much more normal distribution than the Artificial Intelligence community representation. We may be able to attribute this phenomenon to the fact that Gene Expression research is a more mature field.

==== Geographic Distribution

Very similar results geographically with gene expression

#image("./figures/GENE_EXPRESSION/map_scatter_output.png")

This, then is a Kernel Density Estimate (KDE) of where citations are located, and colored by year. Again, East Asia, Western Europe, and Northeast United States dominate.

#image("./figures/GENE_EXPRESSION/global_kde_heatmap_by_year.png")

==== Funding Agency

Again, we see large NIH funding.

#image("./figures/GENE_EXPRESSION/top_5_agencies_by_year.png")

==== Interdisciplinarity Analysis


#image("./figures/GENE_EXPRESSION/interdisciplinary_nodes_plot.png")

#image("./figures/GENE_EXPRESSION/jaccard_vs_citations_scatter.png")

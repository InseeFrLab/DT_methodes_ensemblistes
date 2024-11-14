// #import "@preview/mitex:0.2.4": *
#set math.equation(
numbering: "(1)",
supplement: none
)

#set page("a4")

#set text(font: "IBM Plex Sans")

// Bold titles.
#show table.cell.where(y: 0): set text(weight: "bold")

// Tableaux alignés à gauche, sauf première ligne centrée
#show table.cell: set align(left+horizon)
#show table.cell.where(y: 0): set align(center+horizon)

#show figure.where(
  kind: table
): set figure.caption(position: top)

// Tableau zébré
#set table(
  fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },
  stroke: 0.5pt + rgb("666675"),
)
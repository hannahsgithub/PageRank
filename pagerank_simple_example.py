"""
PageRank Animation — SYDE 312 Project
======================================
Prerequisites:
    pip install manim

Render commands:
    manim -pql pagerank.py PageRankAnimation   # low quality, fast preview
    manim -pqm pagerank.py PageRankAnimation   # medium quality
    manim -pqh pagerank.py PageRankAnimation   # 1080p final render

Output lands in ./media/videos/pagerank/
"""

from manim import *
import numpy as np

# ── palette ───────────────────────────────────────────────────────────────────
NAVY   = "#1a2744"
CTEAL  = "#00b4d8"
CAMBER = "#f4a261"
SMOKE  = "#c8c8c8"
CPURP  = "#7b5ea7"
NODE_R = 0.38   # circle radius for graph nodes


class PageRankAnimation(Scene):

    # ── shared data ────────────────────────────────────────────────────────────
    NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"]

    EDGES = [
        ("P1", "P2"), ("P1", "P4"),
        ("P2", "P3"), ("P2", "P5"),
        ("P3", "P6"),
        ("P4", "P2"), ("P4", "P5"),
        ("P5", "P6"),
        ("P6", "P3"),
    ]

    # Scene-coordinate positions for the 6-node graph
    RAW_POS = {
        "P1": np.array([-5.0,  0.0, 0]),
        "P2": np.array([-1.5,  2.1, 0]),
        "P3": np.array([ 2.5,  2.1, 0]),
        "P4": np.array([-1.5, -2.1, 0]),
        "P5": np.array([ 0.5, -0.5, 0]),
        "P6": np.array([ 2.5, -1.5, 0]),
    }

    NODE_COLORS = {
        "P1": ORANGE,
        "P2": TEAL, "P3": TEAL,
        "P4": BLUE, "P5": BLUE,
        "P6": CPURP,
    }

    # Stochastic transition matrix M (column = source page)
    M = np.array([
        [0,   0,   0,   0,   0, 0  ],
        [0.5, 0,   0,   0.5, 0, 0  ],
        [0,   0.5, 0,   0,   0, 1  ],
        [0.5, 0,   0,   0,   0, 0  ],
        [0,   0.5, 0,   0.5, 0, 0  ],
        [0,   0,   1,   0,   1, 0  ],
    ])

    D = 0.85   # damping factor

    # Converged PageRank scores with d=0.85
    FINAL_RANKS = np.array([0.044, 0.090, 0.322, 0.044, 0.090, 0.410])

    # ── construct ──────────────────────────────────────────────────────────────
    def construct(self):
        n = len(self.NAMES)
        self.G_mat = self.D * self.M + (1 - self.D) * np.ones((n, n)) / n
        self._circles = {}   # name → Circle
        self._clabels  = {}  # name → Text
        self._cur_pos  = dict(self.RAW_POS)   # live positions

        self.part1_title()
        self.part2_graph()
        self.part3_matrix()
        self.part4_power_iteration()
        self.part5_google_matrix()
        self.part6_final_ranking()
        self.part7_verify_eigenvector()

    # ══════════════════════════════════════════════════════════════════════════
    # Part 1 — Title card                                               ~0:00
    # ══════════════════════════════════════════════════════════════════════════
    def part1_title(self):
        title  = Text("PageRank", font_size=80, color=BLUE)
        sub    = Text("Linear Algebra Behind the Web", font_size=30, color=SMOKE)
        course = Text(
            "SYDE 312  ·  Eigenvectors · Stochastic Matrices · Power Iteration",
            font_size=19, color=GRAY,
        )
        grp = VGroup(title, sub, course).arrange(DOWN, buff=0.5)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2), FadeIn(course, shift=UP * 0.1))
        self.wait(1.5)
        self.play(FadeOut(grp))

    # ══════════════════════════════════════════════════════════════════════════
    # Part 2 — Draw the directed graph                                  ~0:00
    # ══════════════════════════════════════════════════════════════════════════
    def part2_graph(self):
        header = self._hdr("Step 1 — The Web as a Directed Graph")
        self.play(Write(header))

        # Animate nodes one at a time
        for name in self.NAMES:
            c = Circle(
                radius=NODE_R,
                color=self.NODE_COLORS[name],
                fill_color=self.NODE_COLORS[name],
                fill_opacity=0.85,
                stroke_width=2,
            ).move_to(self.RAW_POS[name])
            lbl = Text(name, font_size=22, color=WHITE, weight=BOLD).move_to(self.RAW_POS[name])
            self._circles[name] = c
            self._clabels[name]  = lbl
            self.play(GrowFromCenter(c), Write(lbl), run_time=0.30)

        self.wait(0.3)

        # Animate edges one at a time
        self._edge_arrows = {}
        for src, dst in self.EDGES:
            arr = self._arrow(src, dst, self.RAW_POS, color=SMOKE)
            self._edge_arrows[(src, dst)] = arr
            self.play(GrowArrow(arr), run_time=0.22)

        caption = Text(
            "A link A → B is a vote for B.  Votes from important pages carry more weight.",
            font_size=20, color=SMOKE,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(caption))
        self.wait(2.2)
        self.play(FadeOut(caption), FadeOut(header))

    # ══════════════════════════════════════════════════════════════════════════
    # Part 3 — Transition Matrix M                                      ~0:45
    # ══════════════════════════════════════════════════════════════════════════
    def part3_matrix(self):
        header = self._hdr("Step 2 — Building the Transition Matrix M")
        self.play(Write(header))

        # Shrink & shift graph to the left half
        SC   = 0.52
        SHFT = LEFT * 2.6
        new_pos = {n: self.RAW_POS[n] * SC + SHFT for n in self.NAMES}

        self.play(
            *[self._circles[n].animate.scale(SC).move_to(new_pos[n]) for n in self.NAMES],
            *[self._clabels[n].animate.scale(SC).move_to(new_pos[n]) for n in self.NAMES],
            *[FadeOut(a) for a in self._edge_arrows.values()],
            run_time=0.8,
        )
        self._cur_pos = new_pos

        # Redraw thin arrows at new scale
        small_arrs = VGroup(*[
            self._arrow(s, d, new_pos, color=GRAY_C, sw=1.4, r=0.21)
            for s, d in self.EDGES
        ])
        self.play(FadeIn(small_arrs))

        # ── Matrix M display (manual construction for reliability) ──────────
        M_entries = [
            ["0",    "0",    "0",   "0",    "0",  "0"],
            ["½",    "0",    "0",   "½",    "0",  "0"],
            ["0",    "½",    "0",   "0",    "0",  "1"],
            ["½",    "0",    "0",   "0",    "0",  "0"],
            ["0",    "½",    "0",   "½",    "0",  "0"],
            ["0",    "0",    "1",   "0",    "1",  "0"],
        ]
        cell_w, cell_h = 0.50, 0.40
        origin = np.array([1.2, 1.0, 0])   # top-left of matrix body

        matrix_cells = VGroup()
        for row_i, row in enumerate(M_entries):
            for col_j, val in enumerate(row):
                x = origin[0] + col_j * cell_w
                y = origin[1] - row_i * cell_h
                color = CAMBER if val != "0" else SMOKE
                t = Text(val, font_size=20, color=color).move_to([x, y, 0])
                matrix_cells.add(t)

        # Row & column labels
        row_labels = VGroup(*[
            Text(n, font_size=18, color=CTEAL).move_to(
                [origin[0] - 0.40, origin[1] - i * cell_h, 0]
            )
            for i, n in enumerate(self.NAMES)
        ])
        col_labels = VGroup(*[
            Text(n, font_size=18, color=CTEAL).move_to(
                [origin[0] + j * cell_w, origin[1] + 0.38, 0]
            )
            for j, n in enumerate(self.NAMES)
        ])

        # Brackets
        mat_h = 6 * cell_h
        mat_w = 6 * cell_w
        lbr = Text("[", font_size=90, color=WHITE).move_to(
            [origin[0] - 0.62, origin[1] - mat_h / 2 + cell_h / 2, 0]
        )
        rbr = Text("]", font_size=90, color=WHITE).move_to(
            [origin[0] + mat_w - 0.12, origin[1] - mat_h / 2 + cell_h / 2, 0]
        )
        lbl_M = Text("M =", font_size=34, color=WHITE).move_to(
            [origin[0] - 1.10, origin[1] - mat_h / 2 + cell_h / 2, 0]
        )

        matrix_grp = VGroup(lbl_M, lbr, rbr, matrix_cells, row_labels, col_labels)
        self.play(FadeIn(matrix_grp, shift=LEFT * 0.3), run_time=1.0)

        expl = Text(
            "Divide each column by its out-degree → column sums = 1  (column-stochastic)",
            font_size=19, color=CAMBER,
        ).to_edge(DOWN, buff=0.35)
        self.play(Write(expl))

        # Highlight one column to show the idea
        col1_highlight = VGroup(*[matrix_cells[r * 6 + 0] for r in range(6)])
        self.play(col1_highlight.animate.set_color(YELLOW), run_time=0.5)
        self.wait(0.5)
        self.play(col1_highlight.animate.set_color(CAMBER), run_time=0.4)

        self.wait(2.0)

        self._small_arrs = small_arrs
        self._matrix_grp = matrix_grp
        self.play(FadeOut(expl), FadeOut(header),
                  FadeOut(matrix_grp), FadeOut(small_arrs))

    # ══════════════════════════════════════════════════════════════════════════
    # Part 4 — Power Iteration                                          ~1:30
    # ══════════════════════════════════════════════════════════════════════════
    def part4_power_iteration(self):
        header = self._hdr("Step 3 — Power Iteration")
        eq     = Text("v(k+1) = M · v(k)", font_size=32, color=CTEAL)
        eq.next_to(header, DOWN, buff=0.2)
        self.play(Write(header), Write(eq))

        # Reposition nodes compactly left-center
        SC   = 0.60
        SHFT = LEFT * 2.2
        new_pos = {n: self.RAW_POS[n] * SC + SHFT for n in self.NAMES}

        self.play(
            *[self._circles[n].animate.move_to(new_pos[n]) for n in self.NAMES],
            *[self._clabels[n].animate.move_to(new_pos[n]) for n in self.NAMES],
            run_time=0.6,
        )
        self._cur_pos = new_pos

        # Redraw edges
        iter_arrs = VGroup(*[
            self._arrow(s, d, new_pos, color=GRAY_C, sw=1.2, r=0.24)
            for s, d in self.EDGES
        ])
        self.play(FadeIn(iter_arrs))

        # Pre-compute 10 iterations (undamped M to show rank sink)
        n   = 6
        v   = np.ones(n) / n
        iters = [v.copy()]
        for _ in range(10):
            v = self.M @ v
            iters.append(v.copy())

        # Live bar chart on the right
        BAR_ORIGIN = np.array([1.2, -1.8, 0])
        BAR_W      = 0.55
        BAR_GAP    = 0.30
        BAR_MAX_H  = 3.5

        def make_bars(ranks):
            grp = VGroup()
            mx = max(ranks) if max(ranks) > 1e-9 else 1
            for i, nm in enumerate(self.NAMES):
                h = max(ranks[i] / mx * BAR_MAX_H, 0.04)
                x = BAR_ORIGIN[0] + i * (BAR_W + BAR_GAP)
                bar = Rectangle(
                    width=BAR_W, height=h,
                    color=BLUE_D, fill_color=BLUE_D, fill_opacity=0.85,
                ).move_to([x, BAR_ORIGIN[1] + h / 2, 0])
                grp.add(bar)
            return grp

        # Axis baseline
        axis_w = 6 * (BAR_W + BAR_GAP) + 0.2
        baseline = Line(
            start=[BAR_ORIGIN[0] - 0.15, BAR_ORIGIN[1], 0],
            end=[BAR_ORIGIN[0] + axis_w, BAR_ORIGIN[1], 0],
            color=WHITE, stroke_width=1.5,
        )
        bar_name_labels = VGroup(*[
            Text(nm, font_size=16, color=SMOKE).move_to([
                BAR_ORIGIN[0] + i * (BAR_W + BAR_GAP),
                BAR_ORIGIN[1] - 0.28,
                0,
            ])
            for i, nm in enumerate(self.NAMES)
        ])

        self.play(Create(baseline), FadeIn(bar_name_labels))

        bars = make_bars(iters[0])
        self.play(FadeIn(bars))

        iter_label = Text(
            "v⁰ = [1/6, 1/6, ...]ᵀ",
            font_size=26, color=CAMBER,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(iter_label))
        self.wait(0.6)

        for k in range(1, 9):
            new_bars = make_bars(iters[k])
            new_lbl  = Text(
                f"v{k} — iteration {k}",
                font_size=26, color=CAMBER,
            ).to_edge(DOWN, buff=0.4)
            self.play(
                Transform(bars, new_bars),
                Transform(iter_label, new_lbl),
                run_time=0.52,
            )
            self.wait(0.12)

        drain = Text(
            "Rank drains into the P3 ↔ P6 cycle — rank sink!",
            font_size=22, color=RED,
        ).to_edge(DOWN, buff=0.4)
        self.play(Transform(iter_label, drain))
        self.wait(2.0)

        self._iter_arrs     = iter_arrs
        self._iter_bars     = bars
        self._iter_lbl      = iter_label
        self._baseline      = baseline
        self._bar_name_lbls = bar_name_labels
        self.play(
            FadeOut(header), FadeOut(eq),
            FadeOut(bars), FadeOut(iter_label),
            FadeOut(baseline), FadeOut(bar_name_labels),
            FadeOut(iter_arrs),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Part 5 — The Google Matrix fix                                    ~2:00
    # ══════════════════════════════════════════════════════════════════════════
    def part5_google_matrix(self):
        # Fade out graph nodes
        self.play(
            *[FadeOut(self._circles[n]) for n in self.NAMES],
            *[FadeOut(self._clabels[n]) for n in self.NAMES],
        )

        header = self._hdr("Step 4 — The Google Matrix G")
        self.play(Write(header))

        formula = Text(
            "G = d · M + (1−d) · (1/n) · E        d ≈ 0.85",
            font_size=34,
        ).next_to(header, DOWN, buff=0.5)
        self.play(Write(formula))

        bullets_text = [
            ("0.85", "With prob d follow a real link  →  governed by M"),
            ("0.15", "With prob 1−d teleport to any random page  →  governed by ¹⁄ₙ E"),
            ("✓",    "Teleportation breaks rank sinks: surfer can always escape a cycle"),
            ("✓",    "Every page gets non-zero rank  →  Perron–Frobenius applies"),
        ]
        bullets = VGroup(*[
            self._bullet(tag, body) for tag, body in bullets_text
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.40)
        bullets.next_to(formula, DOWN, buff=0.55).shift(LEFT * 0.8)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.2), run_time=0.5)
            self.wait(0.25)

        self.wait(2.0)
        self.play(FadeOut(header), FadeOut(formula), FadeOut(bullets))

    # ══════════════════════════════════════════════════════════════════════════
    # Part 6 — Final ranking with damping comparison table              ~3:00
    # ══════════════════════════════════════════════════════════════════════════
    def part6_final_ranking(self):
        header = self._hdr("Step 6 — Final Ranking with Damping")
        subtitle = Text(
            "Running Power Iteration with G instead of M converges to:",
            font_size=22, color=SMOKE,
        ).next_to(header, DOWN, buff=0.20)
        self.play(Write(header), FadeIn(subtitle))

        # ── Table layout ────────────────────────────────────────────────────
        undamped = [0.000, 0.000, 0.500, 0.000, 0.000, 0.500]
        damped   = self.FINAL_RANKS
        interps  = [
            "Only source of inbound rank is teleportation",
            "Receives from P1 and P4, boosted by teleportation",
            "Still high — the cycle keeps it elevated",
            "Only inbound is P1, which is itself low",
            "Receives from P2 and P4",
            "Highest rank — receives from both P3 and P5",
        ]

        COL_X   = [-5.0, -2.6, -0.9,  3.0]   # Page | v* undamped | v* damped | Interpretation
        ROW_Y0  = 1.6
        ROW_DY  = 0.55
        HDR_Y   = ROW_Y0 + 0.45

        # Column headers
        col_hdrs = VGroup(
            Text("Page",          font_size=21, color=CTEAL, weight=BOLD).move_to([COL_X[0], HDR_Y, 0]),
            Text("v* (no damp)",  font_size=21, color=CTEAL, weight=BOLD).move_to([COL_X[1], HDR_Y, 0]),
            Text("v* (d=0.85)",   font_size=21, color=CTEAL, weight=BOLD).move_to([COL_X[2], HDR_Y, 0]),
            Text("Interpretation",font_size=21, color=CTEAL, weight=BOLD).move_to([COL_X[3], HDR_Y, 0]),
        )
        divider = Line(
            [-6.2, HDR_Y - 0.28, 0], [6.2, HDR_Y - 0.28, 0],
            color=GRAY_C, stroke_width=0.8,
        )
        self.play(FadeIn(col_hdrs), Create(divider))

        row_groups = VGroup()
        for i, (nm, ud, d, interp) in enumerate(zip(self.NAMES, undamped, damped, interps)):
            y = ROW_Y0 - i * ROW_DY
            d_color = CAMBER if d > 0.15 else (TEAL if d > 0.07 else SMOKE)
            row = VGroup(
                Text(nm,          font_size=20, color=WHITE ).move_to([COL_X[0], y, 0]),
                Text(f"{ud:.3f}", font_size=20, color=GRAY_C).move_to([COL_X[1], y, 0]),
                Text(f"{d:.3f}",  font_size=20, color=d_color, weight=BOLD).move_to([COL_X[2], y, 0]),
                Text(interp,      font_size=17, color=SMOKE ).move_to([COL_X[3], y, 0]),
            )
            row_groups.add(row)
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.40)

        # Sum line
        sum_line = Line([-6.2, ROW_Y0 - 6 * ROW_DY + 0.10, 0],
                        [ 6.2, ROW_Y0 - 6 * ROW_DY + 0.10, 0],
                        color=GRAY_C, stroke_width=0.8)
        sum_txt = Text(
            "Sum of all scores:  0.044 + 0.090 + 0.322 + 0.044 + 0.090 + 0.410  =  1.000  ✓",
            font_size=19, color=CAMBER,
        ).move_to([0, ROW_Y0 - 6 * ROW_DY - 0.18, 0])
        self.play(Create(sum_line), Write(sum_txt))

        footer = Text(
            "Every page has a positive score.  P6 is the best-connected destination.\n"
            "P3 ranks second because it feeds directly into P6.",
            font_size=19, color=SMOKE,
        ).to_edge(DOWN, buff=0.30)
        self.play(FadeIn(footer))
        self.wait(3.0)

        self.play(
            FadeOut(header), FadeOut(subtitle), FadeOut(col_hdrs),
            FadeOut(divider), FadeOut(row_groups), FadeOut(sum_line),
            FadeOut(sum_txt), FadeOut(footer),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Part 7 — Verify v* is the eigenvector                             ~4:00
    # ══════════════════════════════════════════════════════════════════════════
    def part7_verify_eigenvector(self):
        header = self._hdr("Step 7 — Verify It Is Actually an Eigenvector")
        desc = Text(
            "To confirm v* is the eigenvector of G with λ = 1, check that  G · v* = v*  for at least one page.",
            font_size=21, color=SMOKE,
        ).next_to(header, DOWN, buff=0.25)
        self.play(Write(header), FadeIn(desc))
        self.wait(0.8)

        # ── Check P6 label ──────────────────────────────────────────────────
        check_lbl = Text(
            "Check P6 (damped case)",
            font_size=24, color=CAMBER, weight=BOLD,
        ).move_to([0, 1.8, 0])
        row_desc = Text(
            "Row 6 of M is  [0, 0, 1, 0, 1, 0]  because P6 receives links from P3 and P5.",
            font_size=21, color=SMOKE,
        ).next_to(check_lbl, DOWN, buff=0.22)
        self.play(FadeIn(check_lbl), FadeIn(row_desc))
        self.wait(0.5)

        # ── Step 1: compute M·v* row 6 ──────────────────────────────────────
        step1_title = Text("Step 1 — compute  M · v*  for row 6:", font_size=21, color=CTEAL)\
            .move_to([0, 0.95, 0])
        self.play(FadeIn(step1_title))

        calc_box = RoundedRectangle(
            corner_radius=0.12, width=9.5, height=1.45,
            color=GRAY_C, fill_color=BLACK, fill_opacity=0.55, stroke_width=0.8,
        ).move_to([0, -0.05, 0])
        calc_line1 = Text(
            "0(0.044) + 0(0.090) + 1(0.322) + 0(0.044) + 1(0.090) + 0(0.410)",
            font_size=20, color=WHITE,
        ).move_to([0, 0.22, 0])
        calc_line2 = Text(
            "= 0.322 + 0.090  =  0.412",
            font_size=20, color=CAMBER,
        ).move_to([0, -0.28, 0])
        self.play(FadeIn(calc_box), Write(calc_line1))
        self.play(Write(calc_line2))
        self.wait(0.8)

        # ── Step 2: apply Google Matrix formula ─────────────────────────────
        step2_title = Text("Step 2 — apply the Google Matrix formula:", font_size=21, color=CTEAL)\
            .move_to([0, -1.15, 0])
        self.play(FadeIn(step2_title))

        gm_box = RoundedRectangle(
            corner_radius=0.12, width=9.5, height=2.10,
            color=GRAY_C, fill_color=BLACK, fill_opacity=0.55, stroke_width=0.8,
        ).move_to([0, -2.25, 0])
        gm_lines = VGroup(
            Text("G · v*[P6]  =  d × 0.412  +  (1−d)/n",   font_size=20, color=WHITE),
            Text("            =  0.85 × 0.412  +  0.15/6",  font_size=20, color=WHITE),
            Text("            =  0.350  +  0.025",           font_size=20, color=WHITE),
            Text("            =  0.410",                     font_size=20, color=CAMBER, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([0, -2.25, 0])

        self.play(FadeIn(gm_box))
        for line in gm_lines:
            self.play(Write(line), run_time=0.55)

        self.wait(0.6)

        # ── Confirmation ────────────────────────────────────────────────────
        confirm = Text(
            "And  v*[P6] = 0.410  ✓   —   G · v* = v*  holds.",
            font_size=22, color=GREEN,
        ).to_edge(DOWN, buff=0.55)
        self.play(Write(confirm))
        self.wait(1.5)

        final_note = Text(
            "The vector v* is stable under multiplication by G.\n"
            "It is the eigenvector with λ = 1.  The PageRank scores are valid.",
            font_size=20, color=SMOKE,
        ).to_edge(DOWN, buff=0.20)
        self.play(Transform(confirm, final_note))
        self.wait(3.0)

        # ── Outro ────────────────────────────────────────────────────────────
        self.play(
            FadeOut(header), FadeOut(desc), FadeOut(check_lbl), FadeOut(row_desc),
            FadeOut(step1_title), FadeOut(calc_box), FadeOut(calc_line1), FadeOut(calc_line2),
            FadeOut(step2_title), FadeOut(gm_box), FadeOut(gm_lines), FadeOut(confirm),
        )
        outro = VGroup(
            Text("PageRank", font_size=64, color=BLUE),
            Text("Messy web  →  one eigenvector of one matrix", font_size=28, color=CTEAL),
            Text("SYDE 312 · Linear Algebra", font_size=22, color=GRAY),
        ).arrange(DOWN, buff=0.5)
        self.play(Write(outro[0]))
        self.play(FadeIn(outro[1], shift=UP * 0.3))
        self.play(FadeIn(outro[2]))
        self.wait(3.0)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _hdr(self, text):
        return Text(text, font_size=30, color=BLUE).to_edge(UP, buff=0.30)

    def _arrow(self, src, dst, pos_map, color=WHITE, sw=2.2, r=0.40):
        """Create a directed arrow between two named nodes, not overlapping circles."""
        s = pos_map[src]
        e = pos_map[dst]
        d = e - s
        u = d / np.linalg.norm(d)
        # Scale the clearance radius by the node scale factor baked into pos_map
        scale_factor = np.linalg.norm(pos_map["P2"] - pos_map["P1"]) / \
                       np.linalg.norm(self.RAW_POS["P2"] - self.RAW_POS["P1"])
        cr = NODE_R * scale_factor + 0.04
        return Arrow(
            s + u * cr, e - u * cr,
            buff=0, stroke_width=sw, color=color,
            max_tip_length_to_length_ratio=0.15,
        )

    def _bullet(self, tag, body):
        dot  = Text(f"[{tag}]", font_size=19, color=CTEAL)
        body = Text(body,       font_size=20, color=SMOKE)
        return VGroup(dot, body).arrange(RIGHT, buff=0.28, aligned_edge=UP)
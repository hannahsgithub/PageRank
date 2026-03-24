"""
PageRank Edge Cases — What Breaks at Scale & How Google Fixed It
================================================================
Scenes:
  1  — Happy path (clean convergence)
  2  — One explicit iteration worked out row-by-row
  3  — Dangling nodes (rank leaks)
  4  — Rank sink / spider trap
  5  — Disconnected components (two isolated clusters)
  6  — The Google Matrix fix
  7  — Fix verified: dangling nodes  [SIDE-BY-SIDE: before | after]
  8  — Fix verified: rank sink       [SIDE-BY-SIDE: before | after]
  9  — Fix verified: disconnected    [SIDE-BY-SIDE: oscillating | converges]
  10 — Eigenvector check (G·v*=v*)
  11 — p sensitivity (why 0.15?)

Render:
    manim -pql pagerank_edgecases.py PageRankEdgeCases   # preview
    manim -pqh pagerank_edgecases.py PageRankEdgeCases   # 1080p
"""

from manim import *
import numpy as np

CTEAL  = "#00b4d8"
CAMBER = "#f4a261"
SMOKE  = "#c8c8c8"
CRED   = "#e63946"
CGREEN = "#2dc653"
NODE_R = 0.34
P_DAMP = 0.15


class PageRankEdgeCases(Scene):

    NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"]

    NODE_COLORS = {
        "P1": ORANGE, "P2": TEAL,   "P3": TEAL,
        "P4": BLUE,   "P5": BLUE,   "P6": PINK,
    }

    # Scene-coordinate positions for the 6-node graph
    POS = {
        "P1": np.array([-4.5,  0.0, 0]),
        "P2": np.array([-1.5,  1.9, 0]),
        "P3": np.array([ 2.0,  1.9, 0]),
        "P4": np.array([-1.5, -1.9, 0]),
        "P5": np.array([ 0.5, -0.5, 0]),
        "P6": np.array([ 2.0, -1.5, 0]),
    }

    EDGES = [
        ("P1","P2"),("P1","P4"),
        ("P2","P3"),("P2","P5"),
        ("P3","P6"),
        ("P4","P2"),("P4","P5"),
        ("P5","P6"),
        ("P6","P3"),
    ]

    # Column-stochastic transition matrix
    M = np.array([
        [0,   0,   0,   0,   0, 0],
        [0.5, 0,   0,   0.5, 0, 0],
        [0,   0.5, 0,   0,   0, 1],
        [0.5, 0,   0,   0,   0, 0],
        [0,   0.5, 0,   0.5, 0, 0],
        [0,   0,   1,   0,   1, 0],
    ])

    def construct(self):
        n = len(self.NAMES)
        self.G_mat = (1 - P_DAMP) * self.M + \
                     P_DAMP * np.ones((n, n)) / n

        self.scene1_happy_path()
        self.scene2_one_iteration()
        self.scene3_dangling_nodes()
        self.scene4_rank_sink()
        self.scene5_disconnected()
        self.scene6_google_fix()
        self.scene7_verify_dangling()
        self.scene8_verify_sink_sidebyside()
        self.scene9_verify_disconnected()
        self.scene10_eigenvector_check()
        self.scene11_p_sensitivity()

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _hdr(self, text, color=BLUE):
        return Text(text, font_size=24, color=color).to_edge(UP, buff=0.22)

    def _arrow(self, s, e, color=SMOKE, sw=1.8, scale=1.0):
        d = e - s
        u = d / np.linalg.norm(d)
        r = NODE_R * scale        # use the actual rendered node radius
        return Arrow(
            s + u * (r + 0.06), e - u * (r + 0.09),
            buff=0, stroke_width=sw, color=color,
            max_tip_length_to_length_ratio=0.18,
        )

    def _draw_graph(self, pos_map, edges, scale=1.0):
        circles, labels, arrows = {}, {}, []
        for nm in self.NAMES:
            c = Circle(
                radius=NODE_R * scale,
                color=self.NODE_COLORS[nm],
                fill_color=self.NODE_COLORS[nm],
                fill_opacity=0.88, stroke_width=2,
            ).move_to(pos_map[nm])
            lbl = Text(nm, font_size=int(18 * scale), color=WHITE,
                       weight=BOLD).move_to(pos_map[nm])
            circles[nm], labels[nm] = c, lbl
        for s, d in edges:
            # Pass scale so arrow offsets match the scaled node radius
            arrows.append(self._arrow(pos_map[s], pos_map[d], scale=scale))
        return circles, labels, arrows

    def _fade_graph(self, circles, labels, arrows):
        self.play(
            *[FadeOut(v) for v in circles.values()],
            *[FadeOut(v) for v in labels.values()],
            *[FadeOut(a) for a in arrows],
        )

    def _bar_chart(self, scores, origin,
                   bar_w=0.50, bar_gap=0.24, bar_max_h=2.6, colors=None):
        if colors is None:
            colors = [self.NODE_COLORS[n] for n in self.NAMES]
        mx = max(scores) if max(scores) > 1e-9 else 1.0
        bars, nlbls, slbls = VGroup(), VGroup(), VGroup()
        for i, (nm, sc, col) in enumerate(zip(self.NAMES, scores, colors)):
            h = max(sc / mx * bar_max_h, 0.04)
            x = origin[0] + i * (bar_w + bar_gap)
            bar = Rectangle(
                width=bar_w, height=h, color=col,
                fill_color=col, fill_opacity=0.85,
            ).move_to([x, origin[1] + h / 2, 0])
            bars.add(bar)
            nlbls.add(
                Text(nm, font_size=14, color=SMOKE)
                .move_to([x, origin[1] - 0.22, 0])
            )
            slbls.add(
                Text(f"{sc:.3f}", font_size=13, color=YELLOW)
                .move_to([x, origin[1] + h + 0.19, 0])
            )
        total_w = len(self.NAMES) * (bar_w + bar_gap)
        baseline = Line(
            [origin[0] - 0.12, origin[1], 0],
            [origin[0] + total_w, origin[1], 0],
            color=WHITE, stroke_width=1.5,
        )
        return bars, nlbls, slbls, baseline

    def _power_iter(self, mat, steps=15, v0=None):
        """Power iteration. Pass v0 to start from a non-uniform distribution."""
        n = mat.shape[0]
        if v0 is None:
            v = np.ones(n) / n
        else:
            v = np.array(v0, dtype=float)
            v /= v.sum()
        hist = [v.copy()]
        for _ in range(steps):
            v = mat @ v
            hist.append(v.copy())
        return hist

    def _animate_bars(self, hist, origin, steps=12, **kwargs):
        """Draw initial bars then animate through iterations. Returns objects to fade."""
        bars, nlbls, slbls, baseline = self._bar_chart(hist[0], origin, **kwargs)
        self.play(Create(baseline), FadeIn(nlbls),
                  *[GrowFromEdge(b, DOWN) for b in bars], FadeIn(slbls))
        iter_lbl = Text("v⁰  equal start", font_size=17, color=CAMBER)\
            .next_to(baseline, DOWN, buff=0.36)
        self.play(Write(iter_lbl))

        for k in range(1, steps + 1):
            nb, _, ns, _ = self._bar_chart(hist[k], origin, **kwargs)
            nl = Text(f"v{k}", font_size=17, color=CAMBER)\
                .next_to(baseline, DOWN, buff=0.36)
            self.play(Transform(bars, nb), Transform(slbls, ns),
                      Transform(iter_lbl, nl), run_time=0.34)

        return bars, nlbls, slbls, baseline, iter_lbl

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 1 — Happy path
    # ══════════════════════════════════════════════════════════════════════════
    def scene1_happy_path(self):
        hdr = self._hdr("Scene 1 — The Happy Path")
        sub = Text("Clean graph · power iteration converges smoothly",
                   font_size=16, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        SC, SHFT = 0.52, LEFT * 3.1
        pos = {n: self.POS[n] * SC + SHFT for n in self.NAMES}
        circles, labels, arrows = self._draw_graph(pos, self.EDGES, SC)
        for nm in self.NAMES:
            self.play(GrowFromCenter(circles[nm]), Write(labels[nm]), run_time=0.20)
        for a in arrows:
            self.play(GrowArrow(a), run_time=0.12)

        ORIGIN = np.array([0.9, -1.55, 0])
        hist = self._power_iter(self.M, steps=12)
        objs = self._animate_bars(hist, ORIGIN, steps=10)
        bars, nlbls, slbls, baseline, iter_lbl = objs

        confirm = Text("Bars stabilise  →  v* found  ✓", font_size=18, color=CGREEN)\
            .next_to(baseline, DOWN, buff=0.36)
        self.play(Transform(iter_lbl, confirm))
        self.wait(2.0)
        self.play(*[FadeOut(o) for o in [hdr, sub, baseline, bars, nlbls, slbls, iter_lbl]])
        self._fade_graph(circles, labels, arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 2 — One explicit iteration worked out row-by-row
    # ══════════════════════════════════════════════════════════════════════════
    def scene2_one_iteration(self):
        hdr = self._hdr("Scene 2 — One Iteration Worked Out")
        sub = Text("v¹ = M · v⁰   where   v⁰ = [1/6, …, 1/6]ᵀ",
                   font_size=17, color=CAMBER).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        rows = [
            ("P1", "0·(1/6)×6 = 0",                      "0.000"),
            ("P2", "½·(1/6) + ½·(1/6)",                   "0.167"),
            ("P3", "½·(1/6) + 1·(1/6)",                   "0.250"),
            ("P4", "½·(1/6)",                              "0.083"),
            ("P5", "½·(1/6) + ½·(1/6)",                   "0.167"),
            ("P6", "1·(1/6) + 1·(1/6)",                   "0.333"),
        ]

        row_h   = 0.52
        start_y = 1.45
        col_x   = [-5.6, -0.6, 4.2]

        hdrs_grp = VGroup(
            Text("Page", font_size=15, color=CTEAL, weight=BOLD).move_to([col_x[0], start_y + 0.38, 0]),
            Text("Row calculation  (each entry of v⁰ = 1/6)", font_size=15, color=CTEAL, weight=BOLD)
                .move_to([col_x[1] + 0.3, start_y + 0.38, 0]),
            Text("v¹[i]", font_size=15, color=CTEAL, weight=BOLD).move_to([col_x[2], start_y + 0.38, 0]),
        )
        div = Line([-6.8, start_y + 0.16, 0], [6.8, start_y + 0.16, 0],
                   color=GRAY_C, stroke_width=0.7)
        self.play(FadeIn(hdrs_grp), Create(div))

        row_objs = []
        for i, (page, calc, result) in enumerate(rows):
            y = start_y - i * row_h
            p_lbl = Text(page,   font_size=15, color=WHITE ).move_to([col_x[0], y, 0])
            c_lbl = Text(calc,   font_size=15, color=SMOKE ).move_to([col_x[1] + 0.3, y, 0])
            r_col = CGREEN if float(result) > 0.1 else (CRED if float(result) == 0 else YELLOW)
            r_lbl = Text(result, font_size=15, color=r_col, weight=BOLD).move_to([col_x[2], y, 0])
            self.play(FadeIn(p_lbl), Write(c_lbl), FadeIn(r_lbl), run_time=0.50)
            row_objs += [p_lbl, c_lbl, r_lbl]

        note = Text(
            "P1 drops to 0 immediately — it has no inbound links.",
            font_size=15, color=CAMBER,
        ).to_edge(DOWN, buff=0.28)
        self.play(Write(note))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in [hdr, sub, hdrs_grp, div, note] + row_objs])

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 3 — Dangling nodes
    # ══════════════════════════════════════════════════════════════════════════
    def scene3_dangling_nodes(self):
        hdr = self._hdr("Scene 3 — Dangling Nodes", color=CRED)
        sub = Text("P3 has no outbound links — rank flows in but never out",
                   font_size=16, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        dangle_edges = [e for e in self.EDGES if e[0] != "P3"]
        SC, SHFT = 0.52, LEFT * 3.1
        pos = {n: self.POS[n] * SC + SHFT for n in self.NAMES}
        circles, labels, arrows = self._draw_graph(pos, dangle_edges, SC)
        for nm in self.NAMES:
            self.play(GrowFromCenter(circles[nm]), Write(labels[nm]), run_time=0.20)
        for a in arrows:
            self.play(GrowArrow(a), run_time=0.12)

        ring = Circle(radius=NODE_R * SC + 0.10, color=CRED, stroke_width=2.5)\
            .move_to(pos["P3"])
        dlbl = Text("Dangling!\n(no outlinks)", font_size=13, color=CRED)\
            .next_to(ring, UP, buff=0.16)
        self.play(Create(ring), Write(dlbl))
        self.wait(0.4)

        M_dangle = self.M.copy()
        M_dangle[:, 2] = 0.0

        col_info = Text("Column P3 of M sums to 0  →  stochastic property broken!",
                        font_size=15, color=CRED).to_edge(DOWN, buff=0.55)
        self.play(Write(col_info))
        self.wait(0.6)

        ORIGIN = np.array([0.9, -1.5, 0])
        n = 6
        v = np.ones(n) / n
        v_hist, mass_hist = [v.copy()], [1.0]
        for _ in range(10):
            v = M_dangle @ v
            v_hist.append(v.copy())
            mass_hist.append(v.sum())

        bars, nlbls, slbls, baseline = self._bar_chart(v_hist[0], ORIGIN, bar_max_h=2.4)
        self.play(Create(baseline), FadeIn(nlbls),
                  *[GrowFromEdge(b, DOWN) for b in bars], FadeIn(slbls))
        mass_lbl = Text("Total mass: 1.000", font_size=15, color=CAMBER)\
            .next_to(baseline, DOWN, buff=0.36)
        self.play(FadeOut(col_info), Write(mass_lbl))

        for k in range(1, 10):
            nb, _, ns, _ = self._bar_chart(v_hist[k], ORIGIN, bar_max_h=2.4)
            nm_txt = Text(f"Total mass: {mass_hist[k]:.3f}  ← leaking!",
                          font_size=15, color=CRED).next_to(baseline, DOWN, buff=0.36)
            self.play(Transform(bars, nb), Transform(slbls, ns),
                      Transform(mass_lbl, nm_txt), run_time=0.38)

        warning = Text("Rank leaks from the system — results are meaningless.",
                       font_size=16, color=CRED).to_edge(DOWN, buff=0.22)
        self.play(Transform(mass_lbl, warning))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in
                    [hdr, sub, ring, dlbl, baseline, bars, nlbls, slbls, mass_lbl]])
        self._fade_graph(circles, labels, arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 4 — Rank sink
    # ══════════════════════════════════════════════════════════════════════════
    def scene4_rank_sink(self):
        hdr = self._hdr("Scene 4 — Rank Sinks (Spider Traps)", color=CRED)
        sub = Text("P3 ↔ P6 form a closed cycle — all rank drains into them",
                   font_size=16, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        SC, SHFT = 0.52, LEFT * 3.1
        pos = {n: self.POS[n] * SC + SHFT for n in self.NAMES}
        circles, labels, arrows = self._draw_graph(pos, self.EDGES, SC)
        for nm in self.NAMES:
            self.play(GrowFromCenter(circles[nm]), Write(labels[nm]), run_time=0.20)
        for a in arrows:
            self.play(GrowArrow(a), run_time=0.12)

        for nm in ["P3", "P6"]:
            self.play(circles[nm].animate.set_fill(CRED, opacity=0.9), run_time=0.28)
        sink_lbl = Text("Rank sink", font_size=13, color=CRED)\
            .next_to(circles["P3"], UP, buff=0.16)
        self.play(Write(sink_lbl))
        self.wait(0.4)

        ORIGIN = np.array([0.9, -1.55, 0])
        hist = self._power_iter(self.M, steps=12)
        bars, nlbls, slbls, baseline = self._bar_chart(hist[0], ORIGIN)
        self.play(Create(baseline), FadeIn(nlbls),
                  *[GrowFromEdge(b, DOWN) for b in bars], FadeIn(slbls))
        iter_lbl = Text("v⁰", font_size=16, color=CAMBER)\
            .next_to(baseline, DOWN, buff=0.36)
        self.play(Write(iter_lbl))

        for k in range(1, 13):
            nb, _, ns, _ = self._bar_chart(hist[k], ORIGIN)
            nb[2].set_fill(CRED); nb[5].set_fill(CRED)
            nl = Text(f"v{k}  P1,P2,P4,P5 draining...",
                      font_size=15, color=CRED if k > 4 else CAMBER)\
                .next_to(baseline, DOWN, buff=0.36)
            self.play(Transform(bars, nb), Transform(slbls, ns),
                      Transform(iter_lbl, nl), run_time=0.40)

        flatline = Text("P1=P2=P4=P5=0.000   P3=P6=0.500   Rank sink confirmed.",
                        font_size=15, color=CRED).to_edge(DOWN, buff=0.22)
        self.play(Transform(iter_lbl, flatline))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in
                    [hdr, sub, sink_lbl, baseline, bars, nlbls, slbls, iter_lbl]])
        self._fade_graph(circles, labels, arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 5 — Disconnected components
    # ══════════════════════════════════════════════════════════════════════════
    def scene5_disconnected(self):
        hdr = self._hdr("Scene 5 — Disconnected Components", color=CRED)
        sub = Text("Two isolated clusters — no links cross between them",
                   font_size=16, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        disc_edges = [
            ("P1","P2"), ("P2","P3"), ("P3","P1"),
            ("P4","P5"), ("P5","P6"), ("P6","P4"),
        ]
        pos_disc = {
            "P1": np.array([-4.2,  1.9, 0]),
            "P2": np.array([-2.5,  0.7, 0]),
            "P3": np.array([-4.2, -0.5, 0]),
            "P4": np.array([ 2.5,  1.9, 0]),
            "P5": np.array([ 4.2,  0.7, 0]),
            "P6": np.array([ 2.5, -0.5, 0]),
        }

        circles, labels, arrows = self._draw_graph(pos_disc, disc_edges, scale=1.0)
        for nm in self.NAMES:
            self.play(GrowFromCenter(circles[nm]), Write(labels[nm]), run_time=0.22)
        for a in arrows:
            self.play(GrowArrow(a), run_time=0.14)

        grp_a  = Text("Group A", font_size=17, color=TEAL).move_to([-3.4, 2.9, 0])
        grp_b  = Text("Group B", font_size=17, color=BLUE ).move_to([ 3.4, 2.9, 0])
        no_link = Text("✗  No links between groups", font_size=16, color=CRED)\
            .move_to([0.0, 0.7, 0])
        self.play(FadeIn(grp_a), FadeIn(grp_b))
        self.play(Write(no_link))
        self.wait(0.6)

        M_disc = np.zeros((6, 6))
        M_disc[1,0]=1; M_disc[2,1]=1; M_disc[0,2]=1
        M_disc[4,3]=1; M_disc[5,4]=1; M_disc[3,5]=1

        # Non-uniform start so the 3-cycles visibly oscillate
        v0 = np.array([0.7, 0.1, 0.2, 0.2, 0.6, 0.2])
        hist = self._power_iter(M_disc, steps=14, v0=v0)

        ORIGIN_A = np.array([-5.5, -2.8, 0])
        ORIGIN_B = np.array([ 0.5, -2.8, 0])
        names_a  = ["P1","P2","P3"]
        names_b  = ["P4","P5","P6"]

        def split_bars(scores, origin_a, origin_b):
            sc_a, sc_b = scores[:3], scores[3:]
            cols_a = [self.NODE_COLORS[n] for n in names_a]
            cols_b = [self.NODE_COLORS[n] for n in names_b]
            mx_a = max(sc_a) if max(sc_a) > 1e-9 else 1
            mx_b = max(sc_b) if max(sc_b) > 1e-9 else 1
            bw, bg, bmh = 0.50, 0.28, 1.1
            grp_a_bars, grp_b_bars = VGroup(), VGroup()
            for i, (sc, col) in enumerate(zip(sc_a, cols_a)):
                h = max(sc / mx_a * bmh, 0.04)
                x = origin_a[0] + i * (bw + bg)
                grp_a_bars.add(
                    Rectangle(width=bw, height=h, color=col,
                               fill_color=col, fill_opacity=0.85)
                    .move_to([x, origin_a[1] + h/2, 0])
                )
            for i, (sc, col) in enumerate(zip(sc_b, cols_b)):
                h = max(sc / mx_b * bmh, 0.04)
                x = origin_b[0] + i * (bw + bg)
                grp_b_bars.add(
                    Rectangle(width=bw, height=h, color=col,
                               fill_color=col, fill_opacity=0.85)
                    .move_to([x, origin_b[1] + h/2, 0])
                )
            return grp_a_bars, grp_b_bars

        bw, bg = 0.50, 0.28
        nlbls_a = VGroup(*[
            Text(n, font_size=13, color=SMOKE)
            .move_to([ORIGIN_A[0] + i*(bw+bg), ORIGIN_A[1]-0.22, 0])
            for i, n in enumerate(names_a)
        ])
        nlbls_b = VGroup(*[
            Text(n, font_size=13, color=SMOKE)
            .move_to([ORIGIN_B[0] + i*(bw+bg), ORIGIN_B[1]-0.22, 0])
            for i, n in enumerate(names_b)
        ])
        baseline_a = Line([ORIGIN_A[0]-0.10, ORIGIN_A[1], 0],
                          [ORIGIN_A[0]+3*(bw+bg), ORIGIN_A[1], 0],
                          color=WHITE, stroke_width=1.2)
        baseline_b = Line([ORIGIN_B[0]-0.10, ORIGIN_B[1], 0],
                          [ORIGIN_B[0]+3*(bw+bg), ORIGIN_B[1], 0],
                          color=WHITE, stroke_width=1.2)

        barsA, barsB = split_bars(hist[0], ORIGIN_A, ORIGIN_B)
        self.play(Create(baseline_a), Create(baseline_b),
                  FadeIn(nlbls_a), FadeIn(nlbls_b))
        self.play(*[GrowFromEdge(b, DOWN) for b in barsA],
                  *[GrowFromEdge(b, DOWN) for b in barsB])

        iter_lbl = Text("v⁰  —  groups converge independently",
                        font_size=15, color=CAMBER).to_edge(DOWN, buff=0.30)
        self.play(Write(iter_lbl))

        for k in range(1, 13):
            nbA, nbB = split_bars(hist[k], ORIGIN_A, ORIGIN_B)
            nl = Text(f"v{k}  —  groups never interact",
                      font_size=15, color=CRED if k > 5 else CAMBER)\
                .to_edge(DOWN, buff=0.30)
            self.play(Transform(barsA, nbA), Transform(barsB, nbB),
                      Transform(iter_lbl, nl), run_time=0.36)

        problem = Text(
            "Rankings are local to each group — comparing P1 vs P4 is meaningless.",
            font_size=15, color=CRED,
        ).to_edge(DOWN, buff=0.22)
        self.play(Transform(iter_lbl, problem))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in
                    [hdr, sub, grp_a, grp_b, no_link,
                     baseline_a, baseline_b, barsA, barsB,
                     nlbls_a, nlbls_b, iter_lbl]])
        self._fade_graph(circles, labels, arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 6 — Google Matrix fix
    # ══════════════════════════════════════════════════════════════════════════
    def scene6_google_fix(self):
        hdr = self._hdr("Scene 6 — The Google Matrix Fix", color=CGREEN)
        formula = Text(
            "G  =  (1 − p) · M  +  p · (1/n) · E        p = 0.15",
            font_size=25, color=WHITE,
        ).next_to(hdr, DOWN, buff=0.38)
        self.play(Write(hdr), Write(formula))

        bullets_data = [
            ("0.85", "Follow a real link  →  governed by M"),
            ("0.15", "Teleport to any random page  →  (1/n)·E"),
            ("✓",    "Every G column sums to 1  →  stochastic restored"),
            ("✓",    "No page can receive zero rank  →  Perron-Frobenius holds"),
            ("✓",    "Teleportation bridges disconnected components globally"),
        ]
        bullets = VGroup(*[
            VGroup(
                Text(f"[{t}]", font_size=16, color=CTEAL),
                Text(b, font_size=16, color=SMOKE),
            ).arrange(RIGHT, buff=0.20, aligned_edge=UP)
            for t, b in bullets_data
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.28)\
          .next_to(formula, DOWN, buff=0.34).shift(LEFT * 0.3)

        for b in bullets:
            self.play(FadeIn(b, shift=RIGHT * 0.18), run_time=0.42)
            self.wait(0.14)
        self.wait(0.6)

        transition = Text("Re-running on the rank-sink graph with G:",
                          font_size=16, color=CAMBER).to_edge(DOWN, buff=0.40)
        self.play(Write(transition))
        self.wait(0.8)
        self.play(FadeOut(hdr), FadeOut(formula), FadeOut(bullets), FadeOut(transition))

        hdr2 = self._hdr("Scene 6 — Power Iteration with G", color=CGREEN)
        self.play(Write(hdr2))

        SC, SHFT = 0.52, LEFT * 3.1
        pos = {n: self.POS[n] * SC + SHFT for n in self.NAMES}
        circles, labels, arrows = self._draw_graph(pos, self.EDGES, SC)
        for nm in self.NAMES:
            self.play(GrowFromCenter(circles[nm]), Write(labels[nm]), run_time=0.18)
        for a in arrows:
            self.play(GrowArrow(a), run_time=0.11)

        ORIGIN = np.array([0.9, -1.55, 0])
        hist_g = self._power_iter(self.G_mat, steps=14)
        objs = self._animate_bars(hist_g, ORIGIN, steps=12)
        bars, nlbls, slbls, baseline, iter_lbl = objs

        confirm = Text(
            "Converged  ✓   P3=0.322  P6=0.410  P2=P5=0.090  P1=P4=0.044",
            font_size=15, color=CGREEN,
        ).next_to(baseline, DOWN, buff=0.36)
        self.play(Transform(iter_lbl, confirm))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in
                    [hdr2, baseline, bars, nlbls, slbls, iter_lbl]])
        self._fade_graph(circles, labels, arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 7 — Fix verified: dangling nodes  (SIDE-BY-SIDE)
    # ══════════════════════════════════════════════════════════════════════════
    def scene7_verify_dangling(self):
        hdr = self._hdr("Scene 7 — Fix Verified: Dangling Nodes", color=CGREEN)
        self.play(Write(hdr))

        # Panel titles — clear of bars (bars start at y≈+0.55, title at y=2.40)
        left_lbl  = Text("Before Fix  (M, no teleport)",  font_size=16, color=CRED  )\
            .move_to([-3.2, 2.55, 0])
        right_lbl = Text("After Fix  (Google Matrix G)",  font_size=16, color=CGREEN)\
            .move_to([ 3.1, 2.55, 0])
        self.play(FadeIn(left_lbl), FadeIn(right_lbl))

        M_dangle = self.M.copy()
        M_dangle[:, 2] = 0.0          # P3 dangling
        n = 6
        G_dangle = (1 - P_DAMP) * M_dangle + P_DAMP * np.ones((n, n)) / n

        # Collect histories
        v0 = np.ones(n) / n
        v_hist_m, mass_m = [v0.copy()], [1.0]
        v = v0.copy()
        for _ in range(12):
            v = M_dangle @ v
            v_hist_m.append(v.copy())
            mass_m.append(round(float(v.sum()), 4))

        v_hist_g, mass_g = [v0.copy()], [1.0]
        v = v0.copy()
        for _ in range(12):
            v = G_dangle @ v
            v_hist_g.append(v.copy())
            mass_g.append(round(float(v.sum()), 4))

        ORIGIN_L = np.array([-5.5, -0.8, 0])
        ORIGIN_R = np.array([ 0.8, -0.8, 0])
        bw, bg, bmh = 0.38, 0.18, 2.1

        bL, nL, sL, blL = self._bar_chart(v_hist_m[0], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
        bR, nR, sR, blR = self._bar_chart(v_hist_g[0], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)

        self.play(Create(blL), Create(blR), FadeIn(nL), FadeIn(nR))
        self.play(*[GrowFromEdge(b, DOWN) for b in bL],
                  *[GrowFromEdge(b, DOWN) for b in bR],
                  FadeIn(sL), FadeIn(sR))

        # Mass labels sit just below each baseline
        mass_lbl_l = Text("Mass: 1.0000", font_size=14, color=CAMBER)\
            .next_to(blL, DOWN, buff=0.44)
        mass_lbl_r = Text("Mass: 1.0000  ✓", font_size=14, color=CGREEN)\
            .next_to(blR, DOWN, buff=0.44)
        iter_lbl = Text("v⁰", font_size=15, color=CAMBER).to_edge(DOWN, buff=0.18)
        self.play(Write(mass_lbl_l), Write(mass_lbl_r), Write(iter_lbl))

        for k in range(1, 13):
            nbL, _, nsL, _ = self._bar_chart(v_hist_m[k], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
            nbR, _, nsR, _ = self._bar_chart(v_hist_g[k], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
            nml = Text(f"Mass: {mass_m[k]:.4f}  ← leaking!", font_size=14, color=CRED)\
                .next_to(blL, DOWN, buff=0.44)
            nmr = Text(f"Mass: {mass_g[k]:.4f}  ✓", font_size=14, color=CGREEN)\
                .next_to(blR, DOWN, buff=0.44)
            nl  = Text(f"v{k}", font_size=15, color=CAMBER).to_edge(DOWN, buff=0.18)
            self.play(
                Transform(bL, nbL), Transform(sL, nsL),
                Transform(bR, nbR), Transform(sR, nsR),
                Transform(mass_lbl_l, nml),
                Transform(mass_lbl_r, nmr),
                Transform(iter_lbl, nl),
                run_time=0.40,
            )

        final = Text(
            "LEFT: rank leaks to zero.  RIGHT: mass stays at 1.000 — dangling node neutralised.",
            font_size=15, color=CGREEN,
        ).to_edge(DOWN, buff=0.18)
        self.play(Transform(iter_lbl, final))
        self.wait(2.5)
        self.play(*[FadeOut(o) for o in
                    [hdr, left_lbl, right_lbl,
                     blL, blR, bL, bR, nL, nR, sL, sR,
                     mass_lbl_l, mass_lbl_r, iter_lbl]])

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 8 — Side-by-side: undamped vs damped rank sink
    # ══════════════════════════════════════════════════════════════════════════
    def scene8_verify_sink_sidebyside(self):
        hdr = self._hdr("Scene 8 — Fix Verified: Rank Sink", color=CGREEN)
        self.play(Write(hdr))

        left_lbl  = Text("Before Fix  (M, p = 0)",        font_size=16, color=CRED  )\
            .move_to([-3.2, 2.55, 0])
        right_lbl = Text("After Fix  (G, p = 0.15)",      font_size=16, color=CGREEN)\
            .move_to([ 3.1, 2.55, 0])
        self.play(FadeIn(left_lbl), FadeIn(right_lbl))

        ORIGIN_L = np.array([-5.5, -0.8, 0])
        ORIGIN_R = np.array([ 0.8, -0.8, 0])
        bw, bg, bmh = 0.38, 0.18, 2.1

        hist_m = self._power_iter(self.M,     steps=14)
        hist_g = self._power_iter(self.G_mat, steps=14)

        def sink_colors(scores):
            return [
                CRED if nm in ["P3","P6"] else
                (GRAY_C if scores[i] < 0.01 else self.NODE_COLORS[nm])
                for i, nm in enumerate(self.NAMES)
            ]

        bL, nL, sL, blL = self._bar_chart(
            hist_m[0], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
        bR, nR, sR, blR = self._bar_chart(
            hist_g[0], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)

        self.play(Create(blL), Create(blR), FadeIn(nL), FadeIn(nR))
        self.play(*[GrowFromEdge(b, DOWN) for b in bL],
                  *[GrowFromEdge(b, DOWN) for b in bR],
                  FadeIn(sL), FadeIn(sR))

        iter_lbl = Text("v⁰", font_size=15, color=CAMBER)\
            .to_edge(DOWN, buff=0.18)
        self.play(Write(iter_lbl))

        for k in range(1, 15):
            nbL, _, nsL, _ = self._bar_chart(
                hist_m[k], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh,
                colors=sink_colors(hist_m[k]))
            nbR, _, nsR, _ = self._bar_chart(
                hist_g[k], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
            nl = Text(
                f"v{k}  |  LEFT: dead pages drain  ·  RIGHT: all pages stay positive",
                font_size=14, color=SMOKE,
            ).to_edge(DOWN, buff=0.18)
            self.play(Transform(bL, nbL), Transform(sL, nsL),
                      Transform(bR, nbR), Transform(sR, nsR),
                      Transform(iter_lbl, nl), run_time=0.40)

        compare = VGroup(
            Text("P1: 0.000 undamped  →  0.044 damped", font_size=14, color=WHITE),
            Text("P2: 0.000 undamped  →  0.090 damped", font_size=14, color=WHITE),
            Text("Teleportation floor p·(1/n)·E prevents any page reaching zero.",
                 font_size=14, color=CAMBER),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)\
         .to_edge(DOWN, buff=0.18)
        self.play(Transform(iter_lbl, compare))
        self.wait(3.0)
        self.play(*[FadeOut(o) for o in
                    [hdr, left_lbl, right_lbl,
                     blL, blR, bL, bR, nL, nR, sL, sR, iter_lbl]])

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 9 — Fix verified: disconnected components
    # ══════════════════════════════════════════════════════════════════════════
    def scene9_verify_disconnected(self):
        hdr = self._hdr("Scene 9 — Fix Verified: Disconnected Components", color=CGREEN)
        sub = Text("Teleportation creates bridges — a global ranking is now possible",
                   font_size=15, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))

        M_disc = np.zeros((6, 6))
        M_disc[1,0]=1; M_disc[2,1]=1; M_disc[0,2]=1   # Group A: 3-cycle
        M_disc[4,3]=1; M_disc[5,4]=1; M_disc[3,5]=1   # Group B: 3-cycle
        n = 6
        G_disc = (1 - P_DAMP) * M_disc + P_DAMP * np.ones((n, n)) / n

        left_lbl  = Text("Before Fix  (M — oscillates)",  font_size=16, color=CRED  )\
            .move_to([-3.2, 2.55, 0])
        right_lbl = Text("After Fix  (G — converges)",    font_size=16, color=CGREEN)\
            .move_to([ 3.1, 2.55, 0])
        self.play(FadeIn(left_lbl), FadeIn(right_lbl))

        # Non-uniform start → makes the 3-cycles oscillate visibly under M_disc
        # Same v0 for G so the right panel also visibly converges from a skewed state
        v0_osc = np.array([0.70, 0.10, 0.20, 0.15, 0.55, 0.30])
        hist_m = self._power_iter(M_disc, steps=14, v0=v0_osc)
        hist_g = self._power_iter(G_disc, steps=14, v0=v0_osc)

        ORIGIN_L = np.array([-5.5, -0.8, 0])
        ORIGIN_R = np.array([ 0.8, -0.8, 0])
        bw, bg, bmh = 0.38, 0.18, 2.1

        bL, nL, sL, blL = self._bar_chart(
            hist_m[0], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
        bR, nR, sR, blR = self._bar_chart(
            hist_g[0], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)

        self.play(Create(blL), Create(blR), FadeIn(nL), FadeIn(nR))
        self.play(*[GrowFromEdge(b, DOWN) for b in bL],
                  *[GrowFromEdge(b, DOWN) for b in bR],
                  FadeIn(sL), FadeIn(sR))

        iter_lbl = Text("v⁰", font_size=15, color=CAMBER)\
            .to_edge(DOWN, buff=0.18)
        self.play(Write(iter_lbl))

        for k in range(1, 15):
            nbL, _, nsL, _ = self._bar_chart(
                hist_m[k], ORIGIN_L, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
            nbR, _, nsR, _ = self._bar_chart(
                hist_g[k], ORIGIN_R, bar_w=bw, bar_gap=bg, bar_max_h=bmh)
            nl = Text(
                f"v{k}  |  LEFT: oscillates — no cross-group rank  ·  RIGHT: converges globally",
                font_size=13, color=SMOKE,
            ).to_edge(DOWN, buff=0.18)
            self.play(Transform(bL, nbL), Transform(sL, nsL),
                      Transform(bR, nbR), Transform(sR, nsR),
                      Transform(iter_lbl, nl), run_time=0.40)

        confirm = VGroup(
            Text("LEFT: bars oscillate — groups cycle independently, no stable cross-group rank.",
                 font_size=14, color=CRED),
            Text("RIGHT: converges to a single global ranking — G bridges the isolated groups.",
                 font_size=14, color=CGREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_edge(DOWN, buff=0.18)
        self.play(Transform(iter_lbl, confirm))
        self.wait(3.0)
        self.play(*[FadeOut(o) for o in
                    [hdr, sub, left_lbl, right_lbl,
                     blL, blR, bL, bR, nL, nR, sL, sR, iter_lbl]])

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 10 — Eigenvector check
    # ══════════════════════════════════════════════════════════════════════════
    def scene10_eigenvector_check(self):
        hdr = self._hdr("Scene 10 — The Fix is Mathematically Valid", color=CTEAL)
        sub = Text("Verify  G · v* = v*  for P6  →  true eigenvector with λ = 1",
                   font_size=16, color=SMOKE).next_to(hdr, DOWN, buff=0.14)
        self.play(Write(hdr), FadeIn(sub))
        self.wait(0.4)

        # ── Step 1 label ──────────────────────────────────────────── y = 2.20
        s1 = Text("Step 1 — Multiply Row 6 of M by v*:",
                  font_size=18, color=CTEAL).move_to([0, 2.20, 0])
        self.play(Write(s1))

        # ── Box 1: two calc lines, centred at y = 1.20 ────────────
        l1 = Text("0(0.044) + 0(0.090) + 1(0.322) + 0(0.044) + 1(0.090) + 0(0.410)",
                  font_size=16, color=WHITE)
        l2 = Text("= 0.322 + 0.090  =  0.412",
                  font_size=16, color=CAMBER)
        box1_content = VGroup(l1, l2).arrange(DOWN, buff=0.22).move_to([0, 1.20, 0])
        box1 = RoundedRectangle(
            corner_radius=0.12, width=11.2,
            height=box1_content.height + 0.50,
            color=GRAY_C, fill_color=BLACK,
            fill_opacity=0.55, stroke_width=0.8,
        ).move_to(box1_content.get_center())
        self.play(FadeIn(box1), Write(l1))
        self.play(Write(l2))
        self.wait(0.5)

        # ── Step 2 label — sits 0.45 below box1 ──────────────────
        s2 = Text("Step 2 — Apply Google Matrix formula:",
                  font_size=18, color=CTEAL)\
            .next_to(box1, DOWN, buff=0.45)
        self.play(Write(s2))

        # ── Box 2: four lines, sits 0.20 below s2 ────────────────
        glines = VGroup(
            Text("G · v*[P6]  =  (1−p) × 0.412  +  p/n",   font_size=16, color=WHITE),
            Text("            =  0.85 × 0.412  +  0.15/6",  font_size=16, color=WHITE),
            Text("            =  0.350  +  0.025",           font_size=16, color=WHITE),
            Text("            =  0.410",                     font_size=16, color=CGREEN, weight=BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        glines.next_to(s2, DOWN, buff=0.28)
        box2 = RoundedRectangle(
            corner_radius=0.12, width=11.2,
            height=glines.height + 0.50,
            color=GRAY_C, fill_color=BLACK,
            fill_opacity=0.55, stroke_width=0.8,
        ).move_to(glines.get_center())

        self.play(FadeIn(box2))
        for line in glines:
            self.play(Write(line), run_time=0.46)

        confirm = Text("v*[P6] = 0.410  ✓   G·v* = v*  →  Perron-Frobenius restored.",
                       font_size=17, color=CGREEN).to_edge(DOWN, buff=0.24)
        self.play(Write(confirm))
        self.wait(3.0)
        self.play(*[FadeOut(o) for o in
                    [hdr, sub, s1, box1, l1, l2, s2, box2, glines, confirm]])

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 11 — p sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    def scene11_p_sensitivity(self):
        hdr = self._hdr("Scene 11 — Why p = 0.15?  Sensitivity Demo", color=CTEAL)
        self.play(Write(hdr))

        p_vals       = [0.01,                0.15,              0.50]
        labels_txt   = ["p = 0.01",          "p = 0.15  ✓",     "p = 0.50"]
        label_colors = [CRED,                CGREEN,             CAMBER]
        ann_txt      = ["Barely fixes sink",  "Goldilocks value", "All bars near equal"]
        ORIGINS = [
            np.array([-5.0, -1.0, 0]),
            np.array([-0.8, -1.0, 0]),
            np.array([ 3.2, -1.0, 0]),
        ]
        n = 6
        all_objs = []

        for p, origin, lt, lc, at in zip(
                p_vals, ORIGINS, labels_txt, label_colors, ann_txt):
            G_p = (1 - p) * self.M + p * np.ones((n, n)) / n
            v_final = self._power_iter(G_p, steps=30)[-1]

            # Panel label sits above bars at fixed y=2.40
            plbl = Text(lt, font_size=15, color=lc)\
                .move_to([origin[0] + 1.15, 2.40, 0])
            # Annotation sits below bars at fixed y=-3.10
            ann = Text(at, font_size=13, color=lc)\
                .move_to([origin[0] + 1.15, -3.10, 0])
            self.play(Write(plbl))
            all_objs += [plbl, ann]

            bars, nlbls, slbls, baseline = self._bar_chart(
                v_final, origin, bar_w=0.36, bar_gap=0.16, bar_max_h=2.2)
            self.play(Create(baseline), FadeIn(nlbls),
                      *[GrowFromEdge(b, DOWN) for b in bars], FadeIn(slbls),
                      run_time=0.48)
            all_objs += [bars, nlbls, slbls, baseline]

        # Show all annotations together
        for ann in [all_objs[1], all_objs[6], all_objs[11]]:
            self.play(FadeIn(ann), run_time=0.35)

        verdict = Text(
            "p = 0.15 fixes all pathological cases while preserving meaningful rank differences.",
            font_size=16, color=WHITE,
        ).to_edge(DOWN, buff=0.20)
        self.play(Write(verdict))
        self.wait(3.5)

        self.play(*[FadeOut(o) for o in all_objs], FadeOut(hdr), FadeOut(verdict))

        outro = VGroup(
            Text("PageRank at Scale", font_size=52, color=BLUE),
            Text("Dangling nodes  ·  Rank sinks  ·  Disconnected components",
                 font_size=20, color=CTEAL),
            Text("All fixed by  G = (1−p)·M + p·(1/n)·E", font_size=19, color=CAMBER),
            Text("SYDE 312  ·  Linear Algebra", font_size=17, color=GRAY),
        ).arrange(DOWN, buff=0.42)
        self.play(Write(outro[0]))
        self.play(FadeIn(outro[1], shift=UP*0.25))
        self.play(FadeIn(outro[2], shift=UP*0.25))
        self.play(FadeIn(outro[3]))
        self.wait(3.0)
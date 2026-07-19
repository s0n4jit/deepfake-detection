# Design: Deepfake Detection Web App — Visual Direction

**Brief interpretation:** this is a forensic analysis tool, not a consumer app. The audience is your instructor and anyone testing the demo — people who want to trust the result, not be entertained by it. The page's single job: let someone upload a face image and read a clear, credible verdict. Minimal, quiet, precise — closer to lab equipment than a startup landing page.

---

## 1. Design Tokens

### Color
A cool, clinical palette — the kind of neutral you'd find in imaging/analysis software, not a marketing site. One accent color, used only for verdict states, so it carries real meaning instead of decoration.

| Token | Hex | Use |
|---|---|---|
| `--bg` | `#F7F8F9` | Page background — near-white, slightly cool, not stark |
| `--surface` | `#FFFFFF` | Cards, upload panel |
| `--ink` | `#1C2128` | Primary text — near-black, not pure black |
| `--muted` | `#5B6470` | Secondary text, labels, captions |
| `--line` | `#DDE1E6` | Hairline borders, dividers |
| `--signal-real` | `#1B7A4D` | "Real" verdict — muted forest green, not a neon success-green |
| `--signal-fake` | `#B4432F` | "Fake" verdict — muted brick red, not alarm-red |
| `--accent` | `#2C5F7C` | Interactive elements (buttons, active states, the scan-frame) — a desaturated steel blue |

No gradients. No decorative color. Every color on the page either is text, a border, or a verdict — nothing is there "to look nice."

### Typography
Two faces, both chosen to read as precise/technical rather than friendly:

- **Display/headings:** `Inter` (or `IBM Plex Sans` if available) — a clean grotesk, used at restrained sizes. This is not a hero-headline product; headings should feel like section labels in a report, not billboard type.
- **Body:** Same family as headings, regular weight, for consistency — a second display face would add personality this brief doesn't want.
- **Data/utility face:** `IBM Plex Mono` (or `JetBrains Mono`) — used specifically for anything numeric: confidence scores, percentages, model names, timestamps, file names. This is the one deliberate typographic choice: **numbers and technical facts are always monospace**, everything else is not. That distinction alone does a lot of work signaling "this is a measurement, not a claim."

Type scale (restrained, few sizes):
- Page title: 20px / 600 weight
- Section label: 13px / 600 weight / uppercase / letter-spacing 0.04em / `--muted` color
- Body: 15px / 400 weight
- Data (mono): 15px body, 28px for the primary confidence number

### Layout
Single column, generous whitespace, no sidebar, no nav bar with five items — this app does one thing. A centered content column, max-width ~640px, on the plain `--bg`.

```
┌──────────────────────────────────────────┐
│                                            │
│   DEEPFAKE DETECTION            [label]   │
│   ─────────────────────────────────       │
│                                            │
│   ┌──────────────────────────────────┐    │
│   │                                    │    │
│   │        drop image here            │    │
│   │        or click to browse         │    │
│   │                                    │    │
│   └──────────────────────────────────┘    │
│                                            │
│   Model:  ( Classical )( CNN )( Both )    │
│                                            │
│              [ Scan Image ]               │
│                                            │
│   ─────────────────────────────────       │
│                                            │
│   RESULT                                  │
│                                            │
│   ┌──────────────────────────────────┐    │
│   │  [face crop with scan-frame]      │    │
│   │                                    │    │
│   │  FAKE            94.2%             │    │
│   │  classical rf     0.71             │    │
│   │  cnn (resnet18)   0.94             │    │
│   └──────────────────────────────────┘    │
│                                            │
└──────────────────────────────────────────┘
```

No card shadows or elevation stacks — a single 1px `--line` border is enough separation. No rounded-corner-heavy "soft UI" — 4px radius maximum, used sparingly (buttons and the upload zone only).

---

## 2. Signature Element

**The scan-frame.** When a face is detected, draw a thin, precise bounding box (1px, `--accent` color) around the detected face region on the uploaded image — with small corner-bracket marks (like a camera focus reticle) instead of a full rectangle. This is the one visual flourish on the page, and it's grounded directly in what the tool actually does: the classical pipeline literally detects a face region and sub-regions (eyes, nose, mouth) as part of its method.

If time allows on Day 3: when "Both" is selected, show two faint overlays in different weights (solid corners for the primary/higher-confidence model, dashed for the secondary) rather than two full boxes — keeps it legible instead of cluttered.

This single element does the job a hero image or animation would do on a marketing site: it's the one thing that signals "this tool is actually looking at your image," and it costs almost nothing to build (a `<canvas>` or absolutely-positioned div over the `<img>`, using the face-detection coordinates your backend already returns).

---

## 3. Component Notes

- **Upload zone:** dashed `--line` border at rest, solid `--accent` border on drag-over. Plain text instructions, no icon library — a single line of copy is enough.
- **Model selector:** three plain toggle buttons (not a dropdown — this is the one meaningful choice a user makes, make it visible), active state uses `--accent` background with white text, inactive is just an outlined button.
- **Scan button:** solid `--accent` fill, white text, 4px radius. One button, no secondary/ghost button competing for attention.
- **Result panel:** appears below the form (not a modal — keep everything in the natural reading flow of one page). Verdict word (`REAL`/`FAKE`) in the display face at the largest size on the page (20–24px, bold, colored with `--signal-real`/`--signal-fake`), confidence percentage next to it in mono.
- **Per-model breakdown:** small mono-font rows, label left / number right, hairline divider between rows — reads like a spec sheet, not a chat bubble.
- **Error/empty states** (per `rules.md` Section 5): shown in the same result panel position, `--muted` text, no red alarm styling for "no face detected" — that's a normal outcome, not a failure. Reserve `--signal-fake` red only for an actual "fake" verdict, never for UI errors, so the color keeps a single consistent meaning throughout.

---

## 4. Motion

Minimal, and only where it clarifies what's happening:
- On upload: a simple 400ms fade/scale-in for the image preview. Nothing bouncy.
- While scanning: a thin 2px progress line at the top of the result panel (not a spinner icon) — reinforces the "scanning/measuring" feel established by the scan-frame.
- Scan-frame corner brackets: draw in with a quick 200ms stroke animation when the result appears, once, not looping or pulsing.
- Respect `prefers-reduced-motion` — disable all of the above and show states instantly if set.

No hover-lift cards, no parallax, no scroll-triggered reveals — this is a single-viewport utility, not a scrolling narrative page.

---

## 5. Copy Voice

- Plain, direct, no marketing language. "Upload a face image to check it" not "Discover the truth behind every photo."
- Verdicts stated as what they are — measurements, not accusations: **"Fake — 94% confidence"**, not "This image is FAKE!"
- Errors explain what happened and what to do, in a flat, factual tone: "No face detected in this image. Try a clearer, front-facing photo." — never "Oops!" or an apology.
- Model names shown exactly as trained (`Classical (FAST+BRIEF+RF)`, `CNN (ResNet18)`) in the mono face — precise labeling reinforces the "this is a measurement tool" feel.

---

## 6. Explicit Non-Choices (things this design deliberately avoids)

- No cream background + terracotta accent, no dark-mode-with-neon-accent, no broadsheet/newspaper columns — the three generic AI-design defaults. This is a cool near-white/graphite palette instead, closer to imaging software than a landing page.
- No numbered step markers (01 / 02 / 03) — there's no multi-step sequence on this page worth numbering.
- No illustration, no stock photography, no icon set — the scan-frame on the user's own uploaded image is the only graphic element, and it's functional, not decorative.
- No card-shadow-heavy "soft UI" — flat surfaces, hairline borders only.

---

## 7. Accessibility Floor

- All interactive elements (upload zone, model toggles, scan button) have visible keyboard focus rings using `--accent`.
- Color is never the only signal — verdict text always includes the word ("Fake"/"Real"), not just the color.
- Responsive down to a single-column mobile layout at the same max-width behavior — the design already is single-column, so this mostly means comfortable touch-target sizing (min 44px height on buttons/toggles) below 480px.
- Sufficient contrast: `--ink` on `--bg` and `--muted` on `--bg` both meet WCAG AA at body text sizes; verify `--signal-real`/`--signal-fake` against `--surface` meet AA before finalizing.

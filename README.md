# tinosai.github.io

My personal tech blog — a hand-built, framework-free **static site** with a
dark "terminal" theme. Served directly from the repo root by GitHub Pages
(Jekyll disabled via `.nojekyll`).

## Structure

```
content/
  about.md                  # about page source (Markdown)
  posts/
    *.html                  # R Markdown -> pandoc posts (imported as-is)
    2019-09-21-ChainRule.markdown
    .Rmds/                  # original R sources for the pandoc posts (archive)
assets/
  css/terminal.css          # the theme
  js/main.js                # small terminal flourishes (clock, typing)
build.py                    # the static site generator
```

Generated output (committed to the repo root, this is what gets served):
`index.html`, `about.html`, `404.html`, `feed.xml`, `posts/<slug>.html`.

## Building

Requires Python 3 and `pandoc` on the PATH.

```sh
python3 build.py          # regenerate the site
python3 -m http.server    # preview at http://localhost:8000
```

## Adding a post

- **Markdown:** drop a file in `content/posts/`, then add an entry to the
  `POSTS` list in `build.py` (slug, source filename, `kind: "markdown"`,
  title, date, tags) and rebuild.
- **pandoc/R Markdown HTML:** export to `content/posts/`, add an entry with
  `kind: "pandoc"`. The generator extracts the `<body>`, strips pandoc's own
  header/scripts, and wraps it in the theme. Math is rendered by MathJax v3.

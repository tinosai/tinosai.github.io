#!/usr/bin/env python3
"""Static site generator for tino's terminal-themed blog.

Reads sources from content/, renders the home page, about page, each post and
an RSS feed into the repo root using a single dark "terminal" template.
The three R Markdown -> pandoc posts are imported by extracting their <body>;
the Chain Rule post is converted from Markdown with pandoc. Math is rendered
client-side by MathJax v3, so it works in dark mode regardless of source.
"""

import html
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "content")
POSTS_DIR = os.path.join(CONTENT, "posts")

SITE_TITLE = "Tino's AI"
SITE_DESC = "Machine Learning, Artificial Intelligence, Code and Algorithms"
SITE_URL = "https://tinosai.github.io"
AUTHOR = "Fortunato Nucera"
LINKEDIN = "https://www.linkedin.com/in/fortunato-nucera-5b35ba111/"
GITHUB = "https://github.com/tinosai"
EMAIL = "fortunato.nucera@icloud.com"

# Ordered newest-first.
POSTS = [
    {
        "slug": "canonical-correlation-analysis",
        "src": "2022-10-07-CCA.html",
        "kind": "pandoc",
        "title": "Canonical Correlation Analysis",
        "date": "2022-10-07",
        "tags": ["statistics", "linear-algebra"],
    },
    {
        "slug": "hamiltonian-monte-carlo",
        "src": "2022-10-02-Hamiltonian-Monte-Carlo.html",
        "kind": "pandoc",
        "title": "Hamiltonian Monte Carlo",
        "date": "2022-10-02",
        "tags": ["bayesian", "sampling", "mcmc"],
    },
    {
        "slug": "quadratic-approximation",
        "src": "2022-09-23-Quadratic-Approximation.html",
        "kind": "pandoc",
        "title": "Quadratic Approximation for Bayesian Inference",
        "date": "2022-09-23",
        "tags": ["bayesian", "inference"],
    },
    {
        "slug": "chain-rule-fcn",
        "src": "2019-09-21-ChainRule.markdown",
        "kind": "markdown",
        "title": "Chain Rule for Fully Connected Neural Networks",
        "date": "2019-09-20",
        "tags": ["neural-networks", "calculus", "backprop"],
    },
]

# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------
IC_LINKEDIN = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.34V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.13 2.06 2.06 0 0 1 0 4.13zM7.12 20.45H3.55V9h3.57v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.22.79 24 1.77 24h20.45C23.2 24 24 23.22 24 22.27V1.73C24 .77 23.2 0 22.22 0z"/></svg>'
IC_GITHUB = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.38.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.6-4.04-1.6-.55-1.4-1.34-1.77-1.34-1.77-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.5 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.8 5.62-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.7.83.58A12 12 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z"/></svg>'
IC_MAIL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/></svg>'
IC_RSS = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6.18 17.82a2.18 2.18 0 1 1-4.36 0 2.18 2.18 0 0 1 4.36 0zM2 9.36v3.05A8.59 8.59 0 0 1 10.59 21h3.05A11.64 11.64 0 0 0 2 9.36zM2 3v3.05A14.95 14.95 0 0 1 16.95 21H20A18 18 0 0 0 2 3z"/></svg>'

MATHJAX = """  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
    };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""


def page(title, description, body, *, is_home=False):
    full_title = SITE_TITLE if is_home else "%s :: %s" % (title, SITE_TITLE)
    tpl = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>%%TITLE%%</title>
  <meta name="description" content="%%DESC%%">
  <meta name="author" content="%%AUTHOR%%">
  <meta property="og:type" content="website">
  <meta property="og:title" content="%%OGTITLE%%">
  <meta property="og:description" content="%%DESC%%">
  <meta name="theme-color" content="#0a0e0c">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/assets/css/terminal.css">
  <link rel="alternate" type="application/rss+xml" title="%%SITE%%" href="/feed.xml">
%%MATHJAX%%
</head>
<body>
  <div class="scanlines"></div>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="dots"><span class="d1"></span><span class="d2"></span><span class="d3"></span></div>
      <div class="topbar-title"><b>tino@github</b>: ~/blog <span class="sep">&mdash;</span> <span data-clock>--:--:--</span></div>
      <nav>
        <a href="/">home</a>
        <a href="/about.html">about</a>
        <a href="/feed.xml">rss</a>
        <a href="%%GITHUB%%" target="_blank" rel="noopener">github</a>
      </nav>
    </div>
  </header>
%%BODY%%
  <footer class="site-footer">
    <div class="footer-inner">
      <div class="row">
        <a href="%%LINKEDIN%%" target="_blank" rel="noopener">linkedin</a><span class="sep">/</span>
        <a href="%%GITHUB%%" target="_blank" rel="noopener">github</a><span class="sep">/</span>
        <a href="mailto:%%EMAIL%%">email</a><span class="sep">/</span>
        <a href="/feed.xml">rss</a>
      </div>
      <div class="footer-copy"><span class="green">$</span> echo &quot;&copy; %%YEAR%% %%AUTHOR%% &mdash; built from scratch, no frameworks&quot;</div>
    </div>
  </footer>
  <script src="/assets/js/main.js" defer></script>
</body>
</html>
"""
    repl = {
        "%%TITLE%%": html.escape(full_title),
        "%%DESC%%": html.escape(description),
        "%%OGTITLE%%": html.escape(full_title),
        "%%AUTHOR%%": AUTHOR,
        "%%SITE%%": html.escape(SITE_TITLE),
        "%%MATHJAX%%": MATHJAX,
        "%%BODY%%": body,
        "%%GITHUB%%": GITHUB,
        "%%LINKEDIN%%": LINKEDIN,
        "%%EMAIL%%": EMAIL,
        "%%YEAR%%": str(datetime.now().year),
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    return tpl


def prompt(cmd_html):
    return '<p class="cmd"><span class="prompt"></span> %s</p>' % cmd_html


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------
def extract_pandoc_body(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = f.read()
    m = re.search(r"<body[^>]*>(.*)</body>", doc, re.S | re.I)
    body = m.group(1) if m else doc
    # Drop scripts/styles that shipped inside the body.
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S | re.I)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I)
    # Drop pandoc's own title header block (we render our own).
    body = re.sub(r'<div id="header">.*?</div>', "", body, count=1, flags=re.S | re.I)
    return body.strip()


def convert_markdown(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # Strip YAML front matter.
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, count=1, flags=re.S)
    # Strip Liquid tags (e.g. the mathjax include).
    raw = re.sub(r"\{%.*?%\}", "", raw)
    # Drop a leading duplicate H2 that repeats the title.
    raw = re.sub(r"\A\s*##\s+Chain Rule for Fully Connected Neural Networks\s*\n",
                 "", raw, count=1)
    out = subprocess.run(
        ["pandoc", "--mathjax", "-f", "markdown", "-t", "html"],
        input=raw, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def render_post_body(post):
    src = os.path.join(POSTS_DIR, post["src"])
    if post["kind"] == "pandoc":
        article = extract_pandoc_body(src)
    else:
        article = convert_markdown(src)
    nice_date = datetime.strptime(post["date"], "%Y-%m-%d").strftime("%b %-d, %Y")
    tags = "".join('<span class="tag">%s</span>' % html.escape(t) for t in post["tags"])
    body = """  <main class="container">
    %s
    <header class="post-head">
      <h1>%s</h1>
      <div class="post-meta">%s %s</div>
    </header>
    <hr class="divider">
    <article class="article">
%s
    </article>
    <a class="back" href="/">$ cd .. &nbsp;&rarr;&nbsp; back to index</a>
  </main>
""" % (
        prompt('cat <span class="arg">posts/%s.md</span>' % html.escape(post["slug"])),
        html.escape(post["title"]),
        nice_date,
        tags,
        article,
    )
    return body


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def render_home():
    rows = []
    for p in POSTS:
        d = datetime.strptime(p["date"], "%Y-%m-%d").strftime("%Y-%m-%d")
        tags = "".join('<span class="tag">%s</span>' % html.escape(t) for t in p["tags"])
        rows.append(
            '<a class="file" href="/posts/%s.html">'
            '<span class="perm">-rw-r--r--</span>'
            '<span class="date">%s</span>'
            '<span><span class="name">%s</span>'
            '<span class="tags">%s</span></span></a>'
            % (p["slug"], d, html.escape(p["title"]), tags)
        )
    filelist = "\n          ".join(rows)
    cmd = "whoami &amp;&amp; cat bio.txt"
    body = """  <main class="container">
    <section class="hero">
      %s
      <div class="output">
        <h1>Fortunato <span class="accent">Nucera</span></h1>
        <p class="role">// Aerodynamics Engineer @ Honda R&amp;D (Japan) &middot; MSc ML &amp; Data Science @ Imperial College London</p>
        <p class="bio">I write about the mathematics behind <span class="key">machine learning</span>, <span class="key">statistics</span>, and the algorithms worth understanding deeply &mdash; Bayesian inference, MCMC, denominator-layout backprop and friends.</p>
        <div class="links">
          <a class="primary" href="%s" target="_blank" rel="noopener">%s linkedin</a>
          <a href="%s" target="_blank" rel="noopener">%s github</a>
          <a href="mailto:%s">%s email</a>
          <a href="/feed.xml">%s rss</a>
        </div>
      </div>
    </section>

    <section>
      %s
      <div class="filelist">
        <div class="total">total %d</div>
          %s
      </div>
    </section>
  </main>
""" % (
        '<p class="cmd"><span class="prompt"></span> <span data-type="%s">%s</span><span class="cursor"></span></p>' % (cmd, cmd),
        LINKEDIN, IC_LINKEDIN, GITHUB, IC_GITHUB, EMAIL, IC_MAIL, IC_RSS,
        prompt('ls <span class="flag">-la</span> <span class="arg">~/posts</span>'),
        len(POSTS), filelist,
    )
    return page(SITE_TITLE, SITE_DESC, body, is_home=True)


def render_about():
    article = convert_markdown_about(os.path.join(CONTENT, "about.md"))
    body = """  <main class="container">
    %s
    <header class="post-head"><h1>About</h1></header>
    <hr class="divider">
    <article class="article">
%s
    </article>
    <a class="back" href="/">$ cd .. &nbsp;&rarr;&nbsp; back to index</a>
  </main>
""" % (prompt('cat <span class="arg">about.txt</span>'), article)
    return page("About", "About " + AUTHOR, body)


def convert_markdown_about(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, count=1, flags=re.S)
    out = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "html"],
        input=raw, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def render_404():
    body = """  <main class="container">
    %s
    <header class="post-head"><h1>404 &mdash; not found</h1></header>
    <hr class="divider">
    <article class="article">
      <p>The path you requested does not exist on this filesystem.</p>
      <pre><code>$ cd %s
bash: cd: no such file or directory</code></pre>
      <p><a href="/">&rarr; return to ~/blog</a></p>
    </article>
  </main>
""" % (prompt('open <span class="arg">$REQUESTED_PATH</span>'), "${REQUESTED_PATH}")
    return page("404", "Page not found", body)


def render_feed():
    items = []
    for p in POSTS:
        dt = datetime.strptime(p["date"], "%Y-%m-%d").replace(
            hour=12, tzinfo=timezone.utc)
        pub = dt.strftime("%a, %d %b %Y %H:%M:%S +0000")
        link = "%s/posts/%s.html" % (SITE_URL, p["slug"])
        desc = "%s — %s" % (p["title"], ", ".join(p["tags"]))
        items.append(
            "    <item>\n"
            "      <title>%s</title>\n"
            "      <link>%s</link>\n"
            "      <guid>%s</guid>\n"
            "      <pubDate>%s</pubDate>\n"
            "      <description>%s</description>\n"
            "    </item>"
            % (html.escape(p["title"]), link, link, pub, html.escape(desc))
        )
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "  <channel>\n"
        "    <title>%s</title>\n"
        "    <link>%s</link>\n"
        '    <atom:link href="%s/feed.xml" rel="self" type="application/rss+xml"/>\n'
        "    <description>%s</description>\n"
        "    <language>en</language>\n"
        "    <lastBuildDate>%s</lastBuildDate>\n"
        "%s\n"
        "  </channel>\n"
        "</rss>\n"
        % (html.escape(SITE_TITLE), SITE_URL, SITE_URL, html.escape(SITE_DESC),
           now, "\n".join(items))
    )


def write(rel, content):
    path = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("  wrote %s (%d bytes)" % (rel, len(content)))


def main():
    print("Building site...")
    write("index.html", render_home())
    write("about.html", render_about())
    write("404.html", render_404())
    write("feed.xml", render_feed())
    for p in POSTS:
        write("posts/%s.html" % p["slug"], page(p["title"], SITE_DESC, render_post_body(p)))
    print("Done.")


if __name__ == "__main__":
    main()

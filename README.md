# Morning Digest

Every morning at 10:30 UTC (5:30 AM EST, 6:30 AM EDT), GitHub Actions:

1. collects new items from the feeds in `feeds.yaml` (last 48 hours, widening to 7 days on slow days; anything already shown is skipped),
2. has Claude (`claude-haiku-5-5`) pick the 5 best against `taste.md` (4 picks plus 1 wildcard from outside your usual interests), each with a one-line reason to read it,
3. renders one static page and deploys it to GitHub Pages.

The page has 👍/👎 buttons on each pick, with reason tags (too verbose, too shallow, great depth…) and an optional note. **Send today's feedback** opens one pre-filled GitHub issue. Submit it and the next run:

- logs the votes to `state/feedback.jsonl` and closes the issue,
- includes your last ~25 reactions (with reasons) in the ranking prompt,
- shows the ranker each source's like/dislike record once it has 3+ votes,
- about once a week, rewrites `state/learned_taste.md`, a short summary of what your votes show, also fed to the ranker.

Only issues opened by the repo owner are read, so other people can't change your taste profile.

## Files

| File | What it is |
|---|---|
| `feeds.yaml` | Feed list (`name`, `url`, optional `category`) |
| `taste.md` | Your taste profile, written by you. Edit it any time. |
| `digest.py` | Fetch → rank → render |
| `templates/` | The page: Newsreader + Inter type, light and dark themes, follows your system setting |
| `state/` | Committed by the workflow: `seen.json`, `feedback.jsonl`, `learned_taste.md`, `meta.json`, `pages.json` |

## Setup

1. **Push this folder to a GitHub repo.** The repo is public by default, so your feedback is too.
2. **Add your API key.** Go to **Settings → Secrets and variables → Actions → New repository secret** and add `ANTHROPIC_API_KEY`. The model is `claude-haiku-5-5`; set an `ANTHROPIC_MODEL` env var to override it.
3. **Enable Pages.** Go to **Settings → Pages → Build and deployment → Source** and choose **GitHub Actions**.
4. **Allow the workflow to push.** Go to **Settings → Actions → General → Workflow permissions** and choose **Read and write permissions**. This lets it commit `state/`.
5. **Run it once.** Go to **Actions → Daily digest → Run workflow**. The page will be at `https://<you>.github.io/<repo>/`.

Without an API key the digest still builds. It falls back to the 5 newest items, using each one's first sentence as the summary.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Offline smoke test: synthetic feeds, a fake LLM response, and no state writes.

```bash
python digest.py --mock && open public/index.html
```

Real feeds and a real LLM, without touching `state/` or GitHub issues:

```bash
ANTHROPIC_API_KEY=sk-ant-... python digest.py --dry-run
```

`--no-llm` forces the chronological fallback. To test the feedback button locally, set `DIGEST_REPO=you/repo` so it opens an issue on that repo. Without it, the button copies the JSON to your clipboard.

## Tuning

- **Change what you want:** edit `taste.md`. It's the top-priority input to the ranker.
- **Add or remove sources:** edit `feeds.yaml` (136 sources, grouped by topic). Feeds tagged `wildcard` exist mainly to supply the wildcard slot. Each source contributes at most its 5 newest items per day, so busy feeds can't crowd out quiet blogs.
- **A blog with no RSS feed:** add it with `type: page` and a `link_pattern` regex matching its post URLs (see the Rikiya Takehi entry). The digest watches the index page for new links and reads each new post's title, date and description once.
- **Window:** set the `LOOKBACK_HOURS` env var (default 48).

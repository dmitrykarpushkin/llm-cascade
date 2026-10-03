"""Default prompts for a daily news digest. Override them through CascadeConfig."""

FILTER_PROMPT = """You filter a news feed for a reader with these interests, in priority order:
{interests}

Below are posts from the last day, each numbered #N. Return ONLY a JSON array (no text around it) with one
object per post worth knowing:
{{"n": N, "topic": "{topics}", "score": 1-5,
  "soon": "date and event if the post is about something important coming up, else null",
  "gist": "the point in one sentence, with numbers"}}

Score: 5 moves markets or is a major event / important within days; 4 significant; 3 notable; 2 minor.
Leave out entirely: ads, giveaways, referral links, "easy money", automated pump/listing signals, memes,
motivation, authors' personal posts, and repeats of the same story (keep the best post).

=== POSTS ===
{material}
"""

WRITER_PROMPT = """Write a short daily update from the pre-filtered news candidates below (topic, importance 1-5,
gist, source), so the reader understands what matters in one minute:

Top
3-5 lines, the most important things of the day (by importance, not topic).

{sections}

Coming up
important upcoming events with dates, if any.

Rules: concise (under ~2000 characters), one line per item, do not repeat Top items in sections, merge repeats
from different sources into one item, end each item with the best source link. No intro or outro, no markdown
except dashes. Do not invent anything beyond the candidates. Skip empty sections.

=== CANDIDATES ({n} of {total} posts, {sources} sources) ===
{material}
"""

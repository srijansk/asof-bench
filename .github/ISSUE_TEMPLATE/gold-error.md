---
name: Suspected gold error
about: An item whose gold answer looks wrong
title: "Gold error: <item id>"
---

**Item id** (the `id` in data/v0.1/questions.jsonl):

**The answer you expect:**

**The evidence:** the repository, path and commit that show it (for example `python/peps`, `peps/pep-0008.rst` and a commit id), and what they show there. The item's as-of commit in each repository is its gold record's `as_of_commits` in data/v0.1/gold.jsonl, the last first-parent commit recorded by the end of the item's date; git's own date filters can miss it (`git log --until` reads raw committer dates, which in python/peps give another commit for every date from 2002-01-08 to 2012-09-22 and on five single days that CONTRIBUTING.md's "Reporting problems" names, and `<branch>@{<date>}` reads your clone's reflog), so give the commit you read:

**Anything else:**

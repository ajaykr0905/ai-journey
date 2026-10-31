# Issue tracker: GitHub

Issues, specifications, milestones, and implementation tickets live in this repository's GitHub Issues. Use the `gh` CLI from inside the repository.

- Create, read, update, comment on, label, and close work through `gh issue`.
- Represent milestone tickets as sub-issues when GitHub supports them.
- Record blocking relationships with native GitHub issue dependencies.
- Fall back to explicit `Part of #...` and `Blocked by: #...` references only when the corresponding GitHub feature is unavailable.
- Publishing a specification or ticket means creating a GitHub issue.
- Pull requests are not a triage request surface.
- Infer the repository from its configured GitHub remote.

# Security

Days 0–17 require no passwords, tokens, cookies, cloud keys, or other credentials.
Credentials mentioned in prior chats or local files are never migrated into this
repository.

- Store any future GitHub token outside the repository, using a trusted credential
  manager or GitHub's supported authentication flow.
- Do not paste secrets into source, issues, pull requests, Actions logs, or commits.
- If a token or key is ever exposed, revoke or rotate it immediately; removing it
  from the latest commit is not sufficient because Git history may retain it.
- Keep `.env` local. The committed `.env.example` intentionally contains no keys.

To report a security concern, contact the repository owner privately rather than
opening a public issue containing sensitive information.

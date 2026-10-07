# Uploading the reference to GitHub

Use this standalone package as the repository root. A private repository is a
useful first step for sharing the reference with a collaborator. Before a public
release, add the authors' chosen license, citation and attribution.

## Upload through the GitHub website

1. Extract the prepared review ZIP and open its inner `advg-reference` folder.
   This is the clean upload folder: it contains the README, source,
   configurations, tests and compact verification evidence.
2. Sign in to GitHub and open [New repository](https://github.com/new).
   Choose your account as owner, name the repository `advg-reference`, and
   select **Private** for the initial collaborator review. Leave the options
   for generating a README, `.gitignore` and license unchecked, then create
   the repository. [GitHub's repository creation instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
3. On the empty repository page, select **uploading an existing file**. In an
   existing repository, use **Add file → Upload files**. Drag the **contents of
   the inner `advg-reference` folder**, including `.gitignore`, into the upload
   area. Upload the extracted files, rather than the ZIP or the outer folder,
   so `README.md` and `pyproject.toml` appear at the repository root. Enter
   `Add neural AdvG reference` as the commit message and commit the upload.
   [GitHub's file upload instructions](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
4. Check the repository home page: the README should render immediately, and
   `advg_reference/`, `configs/`, `examples/`, `tests/` and `verification/`
   should be visible. Open the README's links to check that they resolve.
5. To share a private repository owned by your personal account, open
   **Settings → Collaborators → Add people**, select the professor's GitHub
   username and send the invitation. After acceptance, he can clone or
   download the repository and follow the README.
   [GitHub's collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)

## Alternative: upload using Git

After installing Git and setting up GitHub authentication, create the same
empty repository described above. Open a terminal **inside the clean extracted
`advg-reference` folder** and run:

```text
git init -b main
git add .
git commit -m "Add neural AdvG reference"
git remote add origin https://github.com/YOUR-USERNAME/advg-reference.git
git push -u origin main
```

Replace `YOUR-USERNAME` with the repository owner's account. For later changes,
review `git status` and `git diff`, then add, commit and push the intended files.
[GitHub's instructions for uploading local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)

## Prepare another clean archive

From the package directory, use a new output filename:

```text
python prepare_release.py --output ../tmp/advg-reference-0.4.0-readme.zip
```

The helper refuses to overwrite an existing archive. It includes source,
configuration, tests, documentation, requirements and compact verification
records. It excludes run directories, checkpoints, caches, local environments
and the parent research files. Author-supplied `LICENSE` and `CITATION.cff` are
included when present. The archive contains `SOURCE_MANIFEST.json` with per-file
SHA-256 hashes; a neighboring manifest also records the ZIP's hash.

The prepared archive is a snapshot. If you change files or add a license, create
a new archive with a different filename. Preserve earlier archives as records
of their own contents. Archive preparation is local; publishing and collaborator
invitations happen when you perform the GitHub steps above.

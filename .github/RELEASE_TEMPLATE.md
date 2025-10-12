# Release Checklist

Use this template when creating a new release.

## Pre-Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in:
  - [ ] `agent/config.py` (APP_VERSION)
  - [ ] `web/main.py` (APP_VERSION)
  - [ ] `k8s/app/configmap.yaml` (APP_VERSION)
  - [ ] `README.md`
- [ ] Dependencies reviewed and updated if needed
- [ ] Security vulnerabilities checked (`pip-audit` or `safety`)
- [ ] Docker image builds successfully
- [ ] Kubernetes manifests validated

## Creating a Release

### 1. Update Version

```bash
# Update version in all files
export NEW_VERSION="2.1.0"
# Manually update version strings or use sed:
sed -i '' "s/version: \".*\"/version: \"$NEW_VERSION\"/g" k8s/app/configmap.yaml
```

### 2. Update CHANGELOG.md

Add a new section:

```markdown
## [2.1.0] - 2024-01-15

### Added
- New feature X
- Monitoring dashboard improvements

### Changed
- Updated dependency Y to version Z

### Fixed
- Bug fix A
- Performance improvement B

### Security
- Updated library C to fix CVE-XXXX-YYYY
```

### 3. Commit and Tag

```bash
git add .
git commit -m "Release v$NEW_VERSION"
git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
git push origin main
git push origin "v$NEW_VERSION"
```

### 4. GitHub Actions Will:

- ✅ Build Docker images
- ✅ Push to Docker Hub and GitHub Container Registry
- ✅ Create GitHub Release with changelog
- ✅ Attach documentation and manifests
- ✅ Generate release notes

### 5. Manual Steps After Release

- [ ] Announce release in relevant channels
- [ ] Update production deployment
  ```bash
  argocd app set trading-ai --image ghcr.io/fadih/tradeai:v$NEW_VERSION
  argocd app sync trading-ai
  ```
- [ ] Monitor deployment in Grafana
- [ ] Verify all services healthy
- [ ] Update any external documentation

## Post-Release

- [ ] Monitor error rates in Grafana
- [ ] Check logs for issues
- [ ] Verify metrics are reporting correctly
- [ ] Run smoke tests on production

## Hotfix Process

For urgent fixes:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/v2.1.1 v2.1.0

# Make fixes
git add .
git commit -m "Hotfix: description"

# Tag and push
git tag v2.1.1
git push origin hotfix/v2.1.1
git push origin v2.1.1

# Merge back to main
git checkout main
git merge hotfix/v2.1.1
git push origin main
```

## Version Numbering

Follow Semantic Versioning (semver):

- **MAJOR.MINOR.PATCH** (e.g., 2.1.0)
  - **MAJOR**: Breaking changes
  - **MINOR**: New features (backward compatible)
  - **PATCH**: Bug fixes (backward compatible)

Examples:
- `v2.1.0` → `v2.2.0`: New feature added
- `v2.1.0` → `v2.1.1`: Bug fix
- `v2.1.0` → `v3.0.0`: Breaking API change


# Security Guide: API Key Management

This guide outlines best practices for securely managing API keys and sensitive credentials in the LLM Finetuning Studio project.

## 🔒 Security Principles

### Never Commit Secrets
- **NEVER** commit actual API keys, tokens, or secrets to version control
- Always use placeholder values in example files
- Use environment variables for all sensitive configuration

### Use Environment Variables
- Store all sensitive data in environment variables
- Use `.env` files locally (never committed)
- Use secure secret management in production

## 📁 File Structure

```
├── .env.example          # Template with placeholder values (safe to commit)
├── .env                  # Your actual environment variables (NEVER commit)
├── .gitignore           # Ensures .env files are ignored
└── scripts/
    └── setup-stripe.js  # Uses environment variables, not hardcoded keys
```

## 🚀 Initial Setup

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Fill in Your Actual Values
Edit `.env` and replace all placeholder values with your actual API keys:

```bash
# Replace these with your actual values
STRIPE_API_KEY=sk_test_your_actual_stripe_key_here
VITE_MODAL_API_KEY=your_actual_modal_key_here
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

### 3. Verify .gitignore
Ensure `.env` files are in `.gitignore`:
```
.env
.env.local
.env.*.local
```

## 🔧 API Key Formats and Validation

### Stripe Keys
- **Secret Key**: `sk_test_...` (test) or `sk_live_...` (production)
- **Publishable Key**: `pk_test_...` (test) or `pk_live_...` (production)
- **Webhook Secret**: `whsec_...`

### Modal.com Keys
- **API Key**: Platform-specific format
- **Secret ID**: Unique identifier
- **Token**: Authentication token

### HuggingFace Tokens
- **Format**: `hf_...`
- **Scope**: Read or write permissions

## 🛡️ Security Best Practices

### Development Environment
1. Use test/sandbox API keys only
2. Rotate keys regularly
3. Use minimal permissions
4. Monitor API usage

### Production Environment
1. Use production API keys
2. Store secrets in secure vaults (AWS Secrets Manager, HashiCorp Vault, etc.)
3. Use IAM roles when possible
4. Enable API key restrictions and monitoring
5. Implement rate limiting

### Team Collaboration
1. **Password Manager**: Use tools like 1Password, Bitwarden for sharing
2. **Secure Notes**: Share credentials through encrypted channels
3. **Documentation**: Keep track of which keys are used where
4. **Rotation Schedule**: Regularly rotate shared credentials

## 🚨 Security Incident Response

### If API Keys Are Compromised
1. **Immediately revoke** the exposed keys
2. **Generate new keys** in the respective platforms
3. **Update environment variables** with new keys
4. **Review access logs** for unauthorized usage
5. **Notify team members** of the incident

### If Keys Are Accidentally Committed
1. **Revoke keys immediately**
2. **Remove from git history** (use `git filter-branch` or BFG Repo-Cleaner)
3. **Force push** cleaned history
4. **Generate new keys**
5. **Audit repository** for other potential leaks

## 🔍 Monitoring and Auditing

### Regular Security Checks
- [ ] Review `.gitignore` includes all environment files
- [ ] Audit repository for accidentally committed secrets
- [ ] Check API key usage in platform dashboards
- [ ] Verify team members have appropriate access levels

### Automated Security
```bash
# Install git-secrets for pre-commit hooks
npm install --save-dev git-secrets

# Set up pre-commit hooks
npx husky add .husky/pre-commit "npm run security-check"
```

## 📚 Additional Resources

- [GitHub Secret Scanning](https://docs.github.com/code-security/secret-scanning)
- [Stripe API Security](https://stripe.com/docs/api/authentication)
- [OWASP API Security](https://owasp.org/www-project-api-security/)
- [12-Factor App Configuration](https://12factor.net/config)

## 🤝 Contributing

When contributing to this project:
1. Never include actual API keys in pull requests
2. Use placeholder values in any example files
3. Update this security guide if adding new integrations
4. Test your changes with the security checklist above

---

**Remember**: Security is everyone's responsibility. When in doubt, ask the team!
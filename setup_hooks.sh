#!/usr/bin/env bash
# =============================================================================
# Script d'installation des Git Hooks
# Usage : bash setup_hooks.sh
# =============================================================================

set -euo pipefail

echo "Installation du hook pre-commit..."
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

if command -v pre-commit >/dev/null 2>&1; then
    echo "Telechargement des hooks declaratifs..."
    pre-commit install-hooks
fi

echo "Hook pre-commit installe avec succes."
echo "Dependances recommandees :"
echo "  pip install -r requirements.txt"

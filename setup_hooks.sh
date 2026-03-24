#!/bin/bash
# =============================================================================
# Script d'installation des Git Hooks
# Usage : bash setup_hooks.sh
# =============================================================================

echo "Installation des Git Hooks..."

# Copier le hook pre-commit
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

echo "Hook pre-commit installe avec succes !"
echo "Installez aussi les outils de verification :"
echo "  pip install flake8 black detect-secrets"

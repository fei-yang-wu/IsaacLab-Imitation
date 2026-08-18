"""Cluster-submission control plane: plan, submit, status, logs.

Replaces the export-driven bash submission chain for migrated campaigns. A
campaign is declared in one YAML file; ``plan`` resolves it against a typed
cluster profile into frozen artifacts (batch script, resolved env file,
``plan.json`` with a content hash), and ``submit`` only ever executes a
confirmed plan.
"""

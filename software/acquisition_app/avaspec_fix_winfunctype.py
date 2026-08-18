"""
Patched AvaSpec wrapper module.

This alias module exists so application code can explicitly import the
Windows-callback-fixed wrapper by name instead of the original module name.
"""

from avaspec import *  # noqa: F401,F403


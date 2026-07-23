"""Out-of-process bridges from ``dvxr`` to sibling research packages.

These integrations deliberately shell out to standalone tools rather than importing
them, so incompatible dependency pins and package layouts stay isolated. Importing this
package pulls in only the standard library — no torch, pandas, or matplotlib — which
keeps the torch-free honesty audit unaffected.
"""

from dvxr.integrations.glucose_forecasting import (
    GlucoseForecastingBridge,
    GlucoseRunArtifacts,
    SentinelCommandError,
)

__all__ = [
    "GlucoseForecastingBridge",
    "GlucoseRunArtifacts",
    "SentinelCommandError",
]

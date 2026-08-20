use std::time::Duration;

use serde::Deserialize;

pub(crate) fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    let duration = humantime::parse_duration(&value).map_err(serde::de::Error::custom)?;
    if duration.is_zero() {
        return Err(serde::de::Error::custom(
            "duration must be greater than zero",
        ));
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Settings {
        #[serde(deserialize_with = "deserialize")]
        interval: Duration,
    }

    #[test]
    fn parses_compound_human_duration() {
        let settings: Settings = toml::from_str(r#"interval = "1h 30m""#).unwrap();
        assert_eq!(settings.interval, Duration::from_secs(90 * 60));
    }

    #[test]
    fn rejects_zero_and_numeric_values() {
        assert!(toml::from_str::<Settings>(r#"interval = "0s""#).is_err());
        assert!(toml::from_str::<Settings>("interval = 60").is_err());
    }
}

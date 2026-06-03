#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BannerField {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Banner {
    fields: Vec<BannerField>,
}

impl Banner {
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(field) = self.fields.iter_mut().find(|field| field.key == key) {
            field.value = value;
        } else {
            self.fields.push(BannerField { key, value });
        }
    }

    pub fn set_once(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        if self.fields.iter().any(|field| field.key == key) {
            return;
        }
        self.fields.push(BannerField {
            key,
            value: value.into(),
        });
    }

    pub fn fields(&self) -> &[BannerField] {
        &self.fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_once_keeps_first_value() {
        let mut banner = Banner::default();
        banner.set_once("repo", "first");
        banner.set_once("repo", "second");
        assert_eq!(banner.fields()[0].value, "first");
    }

    #[test]
    fn set_replaces_existing_value_without_moving_field() {
        let mut banner = Banner::default();
        banner.set("config", "old");
        banner.set_once("repo", "repo-url");
        banner.set("config", "new");
        assert_eq!(
            banner.fields(),
            &[
                BannerField {
                    key: "config".into(),
                    value: "new".into(),
                },
                BannerField {
                    key: "repo".into(),
                    value: "repo-url".into(),
                },
            ]
        );
    }
}

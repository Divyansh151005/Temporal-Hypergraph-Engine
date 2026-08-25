//! Optional shared-secret authentication.

use std::sync::Arc;

use tonic::service::Interceptor;
use tonic::{Request, Status};

/// Metadata key carrying the API key.
pub const API_KEY_HEADER: &str = "x-api-key";

/// Rejects calls whose `x-api-key` differs from the configured key; passes
/// everything through when no key is configured.
#[derive(Clone, Default)]
pub struct ApiKeyCheck(Option<Arc<[u8]>>);

impl std::fmt::Debug for ApiKeyCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ApiKeyCheck")
            .field(&self.0.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl ApiKeyCheck {
    /// Requires `key` when it is `Some` and non-empty.
    pub fn new(key: Option<&str>) -> Self {
        Self(
            key.filter(|k| !k.is_empty())
                .map(|k| Arc::from(k.as_bytes())),
        )
    }

    /// Whether a key is enforced.
    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl Interceptor for ApiKeyCheck {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(expected) = &self.0 else {
            return Ok(request);
        };
        match request.metadata().get(API_KEY_HEADER) {
            Some(value) if constant_time_eq(value.as_bytes(), expected) => Ok(request),
            Some(_) => Err(Status::unauthenticated("invalid API key")),
            None => Err(Status::unauthenticated(format!(
                "missing {API_KEY_HEADER} metadata"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(check: &mut ApiKeyCheck, key: Option<&str>) -> Result<(), tonic::Code> {
        let mut request = Request::new(());
        if let Some(key) = key {
            request
                .metadata_mut()
                .insert(API_KEY_HEADER, key.parse().unwrap());
        }
        check.call(request).map(|_| ()).map_err(|s| s.code())
    }

    #[test]
    fn disabled_check_allows_everything() {
        let mut check = ApiKeyCheck::new(Some(""));
        assert!(!check.enabled());
        assert_eq!(call(&mut check, None), Ok(()));
        assert_eq!(call(&mut check, Some("anything")), Ok(()));
    }

    #[test]
    fn enabled_check_requires_matching_key() {
        let mut check = ApiKeyCheck::new(Some("s3cret"));
        assert_eq!(call(&mut check, Some("s3cret")), Ok(()));
        assert_eq!(
            call(&mut check, Some("s3cre")),
            Err(tonic::Code::Unauthenticated)
        );
        assert_eq!(call(&mut check, None), Err(tonic::Code::Unauthenticated));
    }
}

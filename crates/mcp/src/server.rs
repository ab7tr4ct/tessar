//! The `rmcp` handler: a thin layer that exposes [`Tessar`] as MCP tools.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};

use crate::api::Tessar;
use crate::args::{AboutArgs, RecallArgs, RememberArgs};

const INSTRUCTIONS: &str = "\
tessar is the user's long-term memory, shared across conversations and projects. \
Call `recall` before you ask the user something they may have told you before, \
and at the start of work on a topic that earlier sessions may have touched. \
Call `remember` when you learn something worth knowing next time. \
Facts about the person go to the `user` space, facts about the current repository \
to the `project` space.";

const REMEMBER_HEAD: &str = "\
Save facts to long-term memory so they outlive this conversation. Call it when you learn \
something worth knowing next time: a preference, a plan, a decision and its reason, a \
convention, a date, a change in someone's circumstances. Do not save what the code or the \
git history already shows.

SPACES. `user` holds facts about the person you work for (preferences, plans, habits, where \
they live, what they own); they hold in every project. `project` (the default) holds facts \
about the current repository: decisions, conventions, architecture, deadlines. The person is \
the entity {key: \"user\", kind: \"person\"}; it lives in the user space and resolves there \
even when you write to the project space. `mentions` lists other entities a fact is about; \
facts automatically mention the person (user space) or the project (project space), so `about` \
finds them.

FACT KINDS. `statement`: one free-form sentence, for decisions and notes. `attribute`: a \
property of an entity that can change (subject + predicate + value), such as where the user \
lives. `event`: something that happened or will happen on a date; needs `when`. An attribute may take \
`when` too: when the thing described happens (a trip date); the value holds from now. Always give \
`text`: one self-contained sentence in the third person (\"User plans a trip to Japan in May \
2027\"). Searches match against it. Write dates as absolute dates, never \"next Friday\".

PREDICATES (attributes only). A predicate names a relation in snake_case, at most 4 words, and \
is universal: it never contains the thing related. `trip_destination` with value \"Japan\", \
not `plans_trip_to_japan`. Reuse a predicate from the list below when one fits. A predicate is \
many-valued (values accumulate) unless it is `single`, which is only for properties where a \
new value always replaces the old one, such as `lives_in` or `job_title`. Writing a new value \
of a single-valued predicate replaces the old one, which stays in history; the result says \
`superseded`. A new predicate needs `predicate_description` (one line: what it means and what \
its value is) and `single: true` if it is single-valued. If memory finds an existing predicate \
with a similar meaning, the call fails and writes no facts: reuse that predicate, or pass \
`synonym_of` to declare yours its synonym, or `force_new_predicate: true` to define it anyway.

All facts of one call are written, or none. (A new predicate that memory accepted stays defined even if the facts then fail.) Put facts that share a source in one call and pass \
that text as `source`.

CURRENT VOCABULARY, name (single|many): description. Call `vocabulary` for the live list.
";

const RECALL: &str = "\
Search memory in natural language and get the best matching facts, best first. Each line reads `[date recorded] [space] fact`, followed by `[current; earlier: \
value (date), ...]` when a property changed over time, a `(past|now|future: date)` tag for \
facts about a date, and `found by:` saying why the fact matched (entity, words, meaning). \
Searches all spaces by default (this repository, the user, and other repositories); each hit shows its space, and `space` narrows it to `project` or `user`. Set `as_of` (YYYY-MM-DD) to see what was true on that day \
instead of now. Use it before asking the user about something they may have already told \
you, and to check a decision or convention before changing it.";

const VOCABULARY: &str = "\
List the predicates `remember` accepts for attributes: `name (single|many): description`, \
with known synonyms. Check it before proposing a new predicate.";

const ABOUT: &str = "\
List everything memory holds about one entity (a person, project, place, thing), oldest \
first: every fact in which it is the subject, the value, a participant or a mention. The \
person you work for is {key: \"user\", kind: \"person\"}. With `history: true` every version \
of a changed fact is listed, not only the current one.";

/// The MCP server: tools over one [`Tessar`].
#[derive(Clone)]
pub struct TessarServer {
    api: Arc<Tessar>,
    tool_router: ToolRouter<Self>,
}

impl TessarServer {
    /// A server over `api`. The vocabulary summary is baked into the
    /// description of `remember`.
    pub fn new(api: Tessar) -> Self {
        let summary = api.vocabulary_summary();
        let mut tool_router = Self::tool_router();
        if let Some(route) = tool_router.map.get_mut("remember") {
            route.attr.description = Some(Cow::Owned(format!("{REMEMBER_HEAD}{summary}")));
        }
        Self {
            api: Arc::new(api),
            tool_router,
        }
    }

    /// Runs a blocking call on the mapping layer off the async workers: the
    /// embedder and the disk can take a while.
    async fn blocking<T, F>(&self, f: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&Tessar) -> Result<T, String> + Send + 'static,
    {
        let api = Arc::clone(&self.api);
        tokio::task::spawn_blocking(move || f(&api))
            .await
            .map_err(|e| format!("internal error: {e}"))?
    }
}

#[tool_router]
impl TessarServer {
    #[tool(description = REMEMBER_HEAD)]
    async fn remember(&self, Parameters(args): Parameters<RememberArgs>) -> Result<String, String> {
        self.blocking(move |api| api.remember(args)).await
    }

    #[tool(description = RECALL)]
    async fn recall(&self, Parameters(args): Parameters<RecallArgs>) -> Result<String, String> {
        self.blocking(move |api| api.recall(args)).await
    }

    #[tool(description = VOCABULARY)]
    async fn vocabulary(&self) -> Result<String, String> {
        self.blocking(|api| Ok(api.vocabulary())).await
    }

    #[tool(description = ABOUT)]
    async fn about(&self, Parameters(args): Parameters<AboutArgs>) -> Result<String, String> {
        self.blocking(move |api| api.about(args)).await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for TessarServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("tessar", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

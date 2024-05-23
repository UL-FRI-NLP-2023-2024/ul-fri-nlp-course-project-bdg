import { KeyValueStore, PlaywrightCrawler } from "crawlee";

interface Dataset {
  ctx: string;
  ctx_link: string;
  upvotes: number;
  content: string;
  author: string;
}

const GLOBS_TO_CRAWL = ["https://forum.finance.si/?t=*"];

// PlaywrightCrawler crawls the web using a headless
// browser controlled by the Playwright library.
// Used for crawling https://forum.finance.si/
const crawler = new PlaywrightCrawler({
  // Process single forum thread
  async requestHandler({ request, page, enqueueLinks, log }) {
    // Check if the current page is a thread list page
    if (request.url.includes("?f=1") && !request.url.includes("?t=")) {
      // Extract pagination links and enqueue them
      await enqueueLinks({
        selector: "a.f3thread",
      });

      // Enqueue links to forum threads
      await enqueueLinks({
        globs: GLOBS_TO_CRAWL,
      });

      return;
    }

    // Get thread info
    const threadInfo = await page.$(".f3_thread_info");
    // Get id of the forum thread i.e. the last part of the URL after t=
    const retreivedThreadId = request.url?.split("t=")[1];
    // Strip threadId to contain only numbers
    const threadIdMatch = retreivedThreadId?.match(/\d+/);
    const threadId = threadIdMatch ? threadIdMatch[0] : null;

    if (!threadId) {
      log.error(`Thread ID not found in URL: ${request.url}`);
      // Get all links to other forum threads
      await enqueueLinks({
        globs: GLOBS_TO_CRAWL,
      });
      return;
    }

    // Get context of thread (link to discussed article)
    const articleLink = await threadInfo?.evaluate(
      (el) => el.children[1].querySelector("a")?.href
    );
    // Get article link
    const discussedArticle = await threadInfo?.evaluate(
      (el) => el.children[1].querySelector("a")?.textContent
    );

    if (!discussedArticle || !articleLink) {
      log.error(`Thread ID ${threadId} does not contain article link or title`);
      await enqueueLinks({
        globs: GLOBS_TO_CRAWL,
      });
      return;
    }

    // Log current progress to console
    log.info(`Processing thread: ${threadId} with title '${discussedArticle}'`);

    // Get every div that starts with id "forumpost.."
    const posts = await page.$$eval(
      "div[id^='forumpost']",
      (posts) =>
        posts.map((post) => {
          // Get all elements with class "f3_ratepost" and get the second one for upvotes
          const upvotes = post.querySelectorAll(".f3_ratepost")[1]?.textContent;
          // Get post author that is inside a element that is inisde span with class "f3post_author"
          const author = post.querySelector(".f3post_author")?.textContent;

          return {
            content: post.querySelector(".f3content")?.textContent,
            upvotes: upvotes ? parseInt(upvotes.trim()) : 0,
            author: author?.trim() ?? "",
          };
        }),
      { timeout: 10000 }
    );

    // Create key-value store for each forum thread
    // Get previous posts from the key-value store to append new posts
    const currentValue: Dataset[] | null = await KeyValueStore.getValue(
      threadId ?? "OUTPUT"
    );

    const newPosts = posts.map((post) => ({
      ctx: discussedArticle,
      ctx_link: articleLink,
      upvotes: post.upvotes,
      content: post.content,
      author: post.author,
    }));

    await KeyValueStore.setValue(threadId ?? "OUTPUT", [
      ...(currentValue ?? []),
      ...newPosts,
    ]);

    // Enqueue pagination links within the thread
    await enqueueLinks({
      selector: ".pagination a",
    });

    await enqueueLinks({
      globs: GLOBS_TO_CRAWL,
    });
  },
  maxConcurrency: 10,
  // Uncomment this option to see the browser window.
  // headless: false,
  // Comment this option to scrape the full website.
  // maxRequestsPerCrawl: 10,
});

// Add first URL to the queue and start the crawl.
await crawler.run(["https://forum.finance.si/?f=1"]);

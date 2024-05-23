import { KeyValueStore, PlaywrightCrawler } from "crawlee";

interface Dataset {
  ctx: string;
  content: string;
  author: string;
  author_level: string;
}

const GLOBS_TO_CRAWL = ["https://alter.si/tema/*"];

// PlaywrightCrawler crawls the web using a headless
// browser controlled by the Playwright library.
// Used for crawling https://forum.finance.si/
const crawler = new PlaywrightCrawler({
  // Process single forum thread
  async requestHandler({ request, page, enqueueLinks, log }) {
    // Get thread info
    const thread = await page.$(".p-title-value");
    // Get title of thread
    const threadInfo = await thread?.evaluate((el) => el.textContent);
    // Thread id is after last dot in URL: https://alter.si/tema/novo-pri-hot-mobil.2498239/
    let threadId = request.url.split(".").pop();

    // Remove everything after the last slash (inlcuding the slash) in threadId
    threadId = threadId?.substring(0, threadId.lastIndexOf("/"));

    log.info(`Thread ID: ${threadId}`);

    // Check if threadId is number
    if (!threadId || isNaN(parseInt(threadId))) {
      log.error(`Thread ID not found in URL: ${request.url}`);
      // Get all links to other forum threads
      await enqueueLinks({
        globs: GLOBS_TO_CRAWL,
      });
      return;
    }

    // Get every div that starts with class "forumpost"
    const posts = await page.$$eval(
      ".message",
      (posts) =>
        posts.map((post) => {
          // Get all elements with class "f3_ratepost" and get the second one for upvotes
          // Get post author that is inside a element that is inisde span with class "f3post_author"
          const author = post.querySelector(".message-name")?.textContent;
          const authorLevel = post.querySelector(".userTitle")?.textContent;

          return {
            content: post.querySelector(".bbWrapper")?.textContent,
            author: author?.trim() ?? "",
            author_level: authorLevel?.trim() ?? "",
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
      ctx: threadInfo,
      content: post.content,
      author: post.author,
      author_level: post.author_level,
    }));

    await KeyValueStore.setValue(threadId ?? "OUTPUT", [
      ...(currentValue ?? []),
      ...newPosts,
    ]);

    // Enqueue pagination links within the thread
    await enqueueLinks({
      selector: ".pageNav-page a",
    });

    await enqueueLinks({
      globs: GLOBS_TO_CRAWL,
    });
  },
  maxConcurrency: 10,
  // Uncomment this option to see the browser window.
  // headless: false,
  // Comment this option to scrape the full website.
  // maxRequestsPerCrawl: 2,
});

// Add first URL to the queue and start the crawl.
await crawler.run(["https://alter.si/forum/Telefonija"]);

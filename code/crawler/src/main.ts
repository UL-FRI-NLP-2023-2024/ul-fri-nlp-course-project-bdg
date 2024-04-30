// For more information, see https://crawlee.dev/
import { KeyValueStore, PlaywrightCrawler } from "crawlee";

// PlaywrightCrawler crawls the web using a headless
// browser controlled by the Playwright library.
const crawler = new PlaywrightCrawler({
  // Process single forum thread
  async requestHandler({ request, page, enqueueLinks, log }) {
    const title = await page.title();
    // Get id of the forum thread
    const urlParts = request.loadedUrl?.split("/");
    // Get part that starts with "t" and is followed by numbers until / + remove #crta at the end of string if exists
    const threadId = urlParts
      ?.find((part) => part.match(/t\d+/))
      ?.replace(/#crta$/, "");

    if (!threadId) {
      log.error(`Thread ID not found in URL: ${request.loadedUrl}`);
      // Get all links to other forum threads
      // Get all links to other forum threads
      await enqueueLinks({
        globs: [
          {
            glob: "https://slo-tech.com/forum/t*",
            userData: { label: "forum" },
          },
        ],
      });
      return;
    }

    // If no page number is present in the URL (ends  like ..../<pageNumber>), add page number 0 to the URL
    if (!request.loadedUrl?.match(/\/\d+$/)) {
      await enqueueLinks({
        urls: [`https://slo-tech.com/forum/${threadId}/0`],
      });
      return;
    }

    log.info(`Processing thread ${threadId} with title: ${title}`);

    // Get posts from the current page
    await page.waitForSelector(".post");

    // Get content of every post from nested div with class "content"
    const posts = await page.$$eval(".post", (posts) =>
      posts.map((post) =>
        // Get text content inside div
        ({
          message: post.querySelector(".content")?.textContent,
          // User is inside post -> h4 -> a
          user: post.querySelector("h4 a")?.textContent,
        })
      )
    );

    // Create key-value store for each forum thread
    // Get previous posts from the key-value store to append new posts
    const currentValue: { message: string; user: string }[] | null =
      await KeyValueStore.getValue(threadId ?? "OUTPUT");
    await KeyValueStore.setValue(threadId ?? "OUTPUT", [
      ...(currentValue ?? []),
      ...posts,
    ]);

    // Check if next page exists in thread
    // Select a with rel="next" attribute
    const nextButton = await page.$("a[rel='next']");
    log.info(`Next button: ${nextButton}`);

    if (nextButton) {
      // Enqueue the next page
      await enqueueLinks({
        selector: "a[rel='next']",
        label: "page",
      });
      return;
    }

    // Get all links to other forum threads
    await enqueueLinks({
      globs: [
        {
          glob: "https://slo-tech.com/forum/t*",
          userData: { label: "forum" },
        },
      ],
    });
  },
  // Comment this option to scrape the full website.
  // maxRequestsPerCrawl: 1000,
  // Uncomment this option to see the browser window.
  // headless: false,
});

// Add first URL to the queue and start the crawl.
await crawler.run(["https://slo-tech.com/forum"]);
